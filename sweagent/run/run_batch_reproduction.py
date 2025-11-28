"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\ # configure instances
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\     # first 50 instances
    --instances.shuffle=True \\  # shuffle instances (with fixed seed)
    --config config/default.yaml \\
    --agent.model.name gpt-4o  # configure model
[/green]

[cyan][bold]=== LOADING INSTANCES ===[/bold][/cyan]

[cyan][bold]From a file[/bold][/cyan] [green]--instances.type file --instances.path /path/to/file[/green].
[cyan][bold]From huggingface[/bold][/cyan] [green]--instances.type huggingface --instances.dataset_name=SWE_Bench_lite --instances.split=dev[/green].

All instance specifications support the [green]filter[/green], [green]slice[/green], and [green]shuffle[/green] options.
With [green]filter[/green], you can select specific instances, e.g., [green]--instances.filter='instance_id_1|instance_id_2'[/green].
"""
from datetime import datetime
import getpass
import json
import logging
import random
import sys
import time
import traceback
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import shutil
from typing import Self

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.live import Live
from swerex.deployment.hooks.status import SetStatusDeploymentHook

from sweagent import TRAJECTORY_DIR
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.agent.models import ReplayModelConfig
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.run_single import RunSingleConfig
from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (
    add_file_handler,
    add_logger_names_to_stream_handlers,
    get_logger,
    register_thread_name,
    remove_file_handler,
    set_stream_handler_levels,
)

from sweagent.run.code_wrapper import wrap_code_with_ast

RETRY_COUNT = 1

class RunBatchConfig(BaseSettings, cli_implicit_flags=False):
    instances: BatchInstanceSourceConfig = Field(description="Instances to run.")
    agent: AgentConfig = Field(description="Agent options.")
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")
    suffix: str = ""
    """Suffix to add to the output directory. Only used if `output_dir` is `DEFAULT`."""
    raise_exceptions: bool = False
    """Raise exceptions instead of skipping instances."""
    redo_existing: bool = False
    """Do not skip instances that already have a trajectory."""
    env_var_path: Path | None = None
    """Path to a .env file to load environment variables from."""
    num_workers: int = Field(default=1)
    """Number of parallel workers to use."""
    random_delay_multiplier: float = 0.3
    """We will wait for a random amount of time between 0 and `random_delay_multiplier`
    times the number of workers at the start of each instance. This is to avoid any
    potential race condition or issues with bottlenecks, e.g., when running on a platform
    with few CPUs that cannot handle the startup of all containers in time.
    """
    progress_bar: bool = True
    """Whether to show a progress bar. Progress bar is never shown for human models.
    Progress bar is always shown for multi-worker runs.
    """

    # pydantic config
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")

    def set_default_output_dir(self) -> None:
        # Needs to be called explicitly, because self._config_files will be setup
        # post-init.
        if self.output_dir == Path("DEFAULT"):
            user_id = getpass.getuser()
            source_id = self.instances.id
            try:
                model_id = self.agent.model.id  # type: ignore[attr-defined]
            except AttributeError:
                model_id = "unknown"
            config_file = getattr(self, "_config_files", ["no_config"])[0]
            if config_file != "no_config":
                config_file = Path(config_file).stem
            suffix = f"__{self.suffix}" if self.suffix else ""
            self.output_dir = TRAJECTORY_DIR / user_id / f"{config_file}__{model_id}___{source_id}{suffix}"

    @model_validator(mode="after")
    def evaluate_and_redo_existing(self) -> Self:
        if not isinstance(self.instances, SWEBenchInstances):
            return self
        if self.instances.evaluate and self.redo_existing:
            msg = (
                "Cannot evaluate and redo existing at the same time. This would cause invalid results, because "
                "after the first merge_preds gives you a preds.json, this file would be submitted to SB-CLI, causing"
                "evaluation of old instances, which could then not be overwritten by the new ones."
            )
            raise ValueError(msg)
        return self


class _BreakLoop(Exception):
    """Used for internal control flow"""


class RunBatch:
    def __init__(
        self,
        instances: list[BatchInstance],
        agent_config: AgentConfig,
        *,
        output_dir: Path = Path("."),
        hooks: list[RunHook] | None = None,
        raise_exceptions: bool = False,
        redo_existing: bool = False,
        num_workers: int = 1,
        progress_bar: bool = True,
        random_delay_multiplier: float = 0.3,
    ):
        """Note: When initializing this class, make sure to add the hooks that are required by your actions.
        See `from_config` for an example.

        Args:
            hooks: If not specified, the default hooks will be used.
            num_workers: Number of parallel workers to use. Default is 1 (sequential execution).
            progress_bar: Whether to show a progress bar. Progress bar is never shown for human models.
                Progress bar is always shown for multi-worker runs.
            random_delay_multiplier: We will wait for a random amount of time between 0 and `random_delay_multiplier`
                times the number of workers at the start of each instance. This is to avoid any
                potential race conditions.
        """
        if self._model_id in ["human", "human_thought"] and num_workers > 1:
            msg = "Cannot run with human model in parallel"
            raise ValueError(msg)

        self.logger = get_logger("swea-run", emoji="🏃")
        add_file_handler(
            output_dir / "run_batch.log",
            id_="progress",
            filter=lambda name: "swea-run" in name or "config" in name,
        )
        self.instances = instances
        self.agent_config = agent_config
        self.output_dir = output_dir
        self._raise_exceptions = raise_exceptions
        self._chooks = CombinedRunHooks()
        self._redo_existing = redo_existing
        self._num_workers = min(num_workers, len(instances))
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(
            num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml"
        )
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier

    @property
    def _model_id(self) -> str:
        try:
            return self.agent_config.model.id  # type: ignore[attr-defined]
        except AttributeError:
            return "unknown"

    @classmethod
    def from_config(cls, config: RunBatchConfig) -> Self:
        load_environment_variables(config.env_var_path)
        config.set_default_output_dir()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "run_batch.config.yaml").write_text(yaml.dump(config.model_dump_json(), indent=2))
        logger = get_logger("run", emoji="🏃")
        logger.debug("Loading instances from %s", f"{config.instances!r}")
        instances = config.instances.get_instance_configs()

        with open("instances.json", "w") as f:
            for instance in instances:
                instance_id = instance.problem_statement.id
                potentially_bugs = instance.problem_statement.extra_fields["potentially_bugs"]
                f.write(f"======={instance_id}: {potentially_bugs}\n")
            
        logger.info("Loaded %d instances", len(instances))
        if not instances:
            msg = (
                "No instances to run. Here are a few things to check:\n"
                "- With huggingface data: Check that you have the right split (test or dev)\n"
                "- Check your filter does not exclude all instances (check the info log messages)"
            )
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(
            instances=instances,
            agent_config=config.agent,
            output_dir=config.output_dir,
            raise_exceptions=config.raise_exceptions,
            redo_existing=config.redo_existing,
            num_workers=config.num_workers,
            progress_bar=config.progress_bar,
            random_delay_multiplier=config.random_delay_multiplier,
        )
        if isinstance(config.instances, SWEBenchInstances) and config.instances.evaluate:
            from sweagent.run.hooks.swe_bench_evaluate import SweBenchEvaluate

            rb.add_hook(
                SweBenchEvaluate(
                    output_dir=config.output_dir,
                    subset=config.instances.subset,
                    split=config.instances.split,
                    continuous_submission_every=30,
                )
            )
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()

        if self._num_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()

        output_dirs = []
        for instance in self.instances:
            output_dirs.append(self.output_dir / instance.problem_statement.id)
        merge_predictions(output_dirs, self.output_dir / "preds.json")

        self._chooks.on_end()

    def main_single_worker(self) -> None:
        with ExitStack() as stack:
            # Conditionally add progress bar
            if self._model_id not in ["human", "human_thought"] and self._show_progress_bar:
                stack.enter_context(Live(self._progress_manager.render_group))
            for instance in self.instances:
                try:
                    for i in range(RETRY_COUNT):
                        self.run_instance(instance)
                except _BreakLoop:
                    self.logger.info("Stopping loop over instances")
                    break

    def main_multi_worker(self) -> None:
        add_logger_names_to_stream_handlers()
        # Set all stream handlers to WARNING and set everything where we want to have
        # more verbosity explicitly
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)  # type: ignore

        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = [executor.submit(self.run_instance, instance) for instance in self.instances]
                try:
                    for future in as_completed(futures):
                        future.result()
                except (KeyboardInterrupt, _BreakLoop):
                    msg = (
                        "Received keyboard interrupt, waiting for running instances "
                        "to finish, but cancelled everything else"
                    )
                    self.logger.info(msg)
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()

    def run_instance(self, instance: BatchInstance) -> None:
        run_id = 0
        out_dir = Path(self.output_dir) / instance.problem_statement.id / f"{run_id}"
        while out_dir.exists():
            run_id += 1
            out_dir = Path(self.output_dir) / instance.problem_statement.id / f"{run_id}"
            
        self.logger.info("Running on instance %s", instance.problem_statement.id)
        register_thread_name(instance.problem_statement.id)
        self._add_instance_log_file_handlers(instance.problem_statement.id, multi_worker=self._num_workers > 1, out_dir=out_dir)
        # Let's add some randomness to avoid any potential race conditions or thundering herd
        if self._progress_manager.n_completed < self._num_workers:
            time.sleep(random.random() * self._random_delay_multiplier * (self._num_workers - 1))

        self._progress_manager.on_instance_start(instance.problem_statement.id)

        try:
            result = self._run_instance(instance, out_dir, run_agent=True)
        except KeyboardInterrupt:
            raise _BreakLoop
        except (SystemExit, ModelConfigurationError, TotalCostLimitExceededError) as e:
            if self._raise_exceptions:
                raise
            self.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"❌ Failed on {instance.problem_statement.id}: {e}")
            self._progress_manager.on_uncaught_exception(instance.problem_statement.id, e)
            if self._raise_exceptions:
                raise
        else:
            self._progress_manager.on_instance_end(
                instance.problem_statement.id, exit_status=result.info.get("exit_status", "unknown_exit")
            )
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance.problem_statement.id)
            
    def _setup_env(self, instance: BatchInstance, output_dir: Path) -> SWEEnv:
        """Set up and start the environment for the given instance."""
        self._progress_manager.update_instance_status(instance.problem_statement.id, "Starting environment")
        instance.env.name = f"{instance.problem_statement.id}"
        env = SWEEnv.from_config(instance.env)
        env.add_hook(
            SetStatusEnvironmentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        env.deployment.add_hook(
            SetStatusDeploymentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        env.start()
        self._chooks.on_instance_start(index=0, env=env, problem_statement=instance.problem_statement)
        return env

    def _run_agent(self, instance: BatchInstance, env: SWEEnv, output_dir: Path) -> AgentRunResult:
        """Run the agent with the provided environment."""
        self.agent_config.name = f"{instance.problem_statement.id}"
        agent = get_agent_from_config(self.agent_config)
        single_run_replay_config = RunSingleConfig(
            agent=self.agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
        (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
        )
        agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
        agent.add_hook(SetStatusAgentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        
        try:
            result = agent.run(
                problem_statement=instance.problem_statement,
                env=env,
                output_dir=output_dir,
            )
        except Exception:
            # The actual handling is happening in `run_instance`, but we need to make sure that
            # we log it to the agent specific logger as well
            agent.logger.error(traceback.format_exc())  # type: ignore[attr-defined]
            raise
        
        save_predictions(self.output_dir, instance.problem_statement.id, result)
        self._chooks.on_instance_completed(result=result)
        return result

    def _run_instance(self, instance: BatchInstance, out_dir: Path, run_agent: bool = True) -> AgentRunResult:
        output_dir = out_dir
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # copy /Documents/summarize_traj/merged_traj_arvo_117/instance_id
        # to the output_dir
        if not output_dir.exists():
            self.logger.info(f"Copying trajectory from /Documents/summarize_traj/merged_traj_arvo_117/instance_id to {output_dir}")
            # only copy the .traj file
            traj_path = Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id / f"{instance.problem_statement.id}.traj"
            if traj_path.exists():
                shutil.copy(
                    Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id / f"{instance.problem_statement.id}.traj",
                    output_dir / f"{instance.problem_statement.id}.traj"
                )
            # shutil.copytree(
            #     Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id,
            #     output_dir
            # )
        else:
            # check if the .trajs file exists
            if not (output_dir / f"{instance.problem_statement.id}.traj").exists():
                self.logger.info(f"Copying trajectory from /Documents/summarize_traj/merged_traj_arvo_117/instance_id to {output_dir}")
                # delete the output_dir
                shutil.rmtree(output_dir)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                traj_path = Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id / f"{instance.problem_statement.id}.traj"
                if traj_path.exists():
                    shutil.copy(
                        Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id / f"{instance.problem_statement.id}.traj",
                        output_dir / f"{instance.problem_statement.id}.traj"
                    )
                # shutil.copytree(
                #     Path("/Documents/summarize_traj/merged_traj_arvo_117") / instance.problem_statement.id,
                #     output_dir
                # )
        # output_dir.mkdir(parents=True, exist_ok=True)
        
        env = self._setup_env(instance, out_dir)
        

        # run the command in the environment to append the text "temp_tests", "reproduce_bug.py", "reproduce_non_bug.py" into the .gitignore file
        env.communicate('echo "temp_tests" >> .gitignore', check="raise")
        env.communicate('echo "reproduce_bug.py" >> .gitignore', check="raise")
        env.communicate('echo "reproduce_non_bug.py" >> .gitignore', check="raise")
        
        # make the unit_tests folder in the environment
        env.communicate('mkdir -p unit_tests', check="raise")
        env.communicate('echo "unit_tests/" >> .gitignore', check="raise")
        
        # copy the unit_tests folder to the environment, from the folder "/data/Documents/test_creator/all_enhanced_tests/{instance_id}/test*.py"
        unit_tests_folder = Path("/data/Documents/test_creator/method3_fixed/all_enhanced_tests") / instance.problem_statement.id
        if unit_tests_folder.exists():
            for file in unit_tests_folder.glob("test*.py"):
                content = file.read_text()
                env.write_file(f"unit_tests/{file.name}", content)
                print(f"Copied {file} to {env.read_file(f'unit_tests/{file.name}')}")
                # also write unit test to output_dir
                # make the unit_tests folder in the output_dir
                (output_dir / f"unit_tests").mkdir(parents=True, exist_ok=True)
                (output_dir / f"unit_tests/{file.name}").write_text(content)
                        
        
        # create the temp_tests folder
        env.communicate('mkdir -p temp_tests', check="raise")
                
        # Use ReplayModel to replay the previous trajectory
        traj_path = output_dir / f"{instance.problem_statement.id}.traj"
        if not traj_path.exists():
        # if True:
            # self.logger.warning(f"No trajectory file found at {traj_path}, creating minimal result")
            # result = AgentRunResult(
            #     info={"exit_status": "no_trajectory"},
            #     trajectory=[],
            # )
            # env.close()
            # return result
            
            try:
                result = self._run_agent(instance, env, output_dir)
            except Exception as e:
                self.logger.error(f"Failed to run agent: {e}")
                self.logger.error(traceback.format_exc())
            finally:
                env.close()
                
            return result
        
        try:
            # Load trajectory data
            traj_data = json.loads(traj_path.read_text())
            
            # Create actions file for ReplayModel
            replay_action_trajs_path = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
            
            # Extract actions from trajectory history
            actions = []
            has_tool_calls = any(
                "tool_calls" in item and item["tool_calls"] is not None
                for item in traj_data["history"]
                if item["role"] == "assistant"
            )
            
            # Check agent configuration for function calling compatibility
            agent_config = self.agent_config
            parse_function = agent_config.tools.parse_function.type
            use_function_calling = parse_function == "function_calling"
            
            if has_tool_calls and not use_function_calling:
                self.logger.warning(
                    "Trajectory contains tool calls but config is not set up for function calling. "
                    "This may cause replay issues."
                )
            
            # Extract actions from history
            for ix, item in enumerate(traj_data["history"]):
                if item["role"] != "assistant":
                    continue
                action = {"message": item["content"]}
                if use_function_calling and "tool_calls" in item and item["tool_calls"] is not None:
                    action["tool_calls"] = item["tool_calls"]
                actions.append(action)
            
            if len(actions) == 0:
                self.logger.warning("No actions found in trajectory")
                result = AgentRunResult(
                    info={"exit_status": "no_actions"},
                    trajectory=[],
                )
                env.close()
                return result
            
            # Save actions file for ReplayModel
            replay_action_trajs_path.write_text(json.dumps({instance.problem_statement.id: actions}))
            
            # Create agent config with ReplayModel
            replay_agent_config = self.agent_config.model_copy(deep=True)
            replay_agent_config.model = ReplayModelConfig(replay_path=replay_action_trajs_path)
            replay_agent_config.name = f"{instance.problem_statement.id}_replay"
            
            # Create and run agent with ReplayModel
            agent = get_agent_from_config(replay_agent_config)
            single_run_replay_config = RunSingleConfig(
                agent=replay_agent_config,
                problem_statement=instance.problem_statement,
                env=instance.env,
            )
            (output_dir / f"{instance.problem_statement.id}.replay.config.yaml").write_text(
                yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            agent.replay_config = single_run_replay_config
            agent.add_hook(SetStatusAgentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
            
            try:
                self.logger.info(f"Replaying trajectory for {instance.problem_statement.id}")
                agent.run(
                    problem_statement=instance.problem_statement,
                    env=env,
                    output_dir=output_dir,
                )
                
                # After replay, get previous predictions and run docker commands
                pred_path = output_dir / f"{instance.problem_statement.id}.pred"
                if pred_path.exists():
                    pred = json.load(pred_path.open())
                    # self.docker_command(pred, env, output_dir, instance)
                else:
                    self.logger.warning(f"No prediction file found at {pred_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to replay trajectory: {e}")
                self.logger.error(traceback.format_exc())
                # Fallback to loading existing trajectory data
                result = AgentRunResult(
                    info=traj_data.get("info", {"exit_status": "replay_failed"}),
                    trajectory=traj_data.get("trajectory", []),
                )
            finally:
                # Clean up temporary file
                try:
                    replay_action_trajs_path.unlink()
                except Exception:
                    pass

            try:
                result = self._run_agent(instance, env, output_dir)
            except Exception as e:
                self.logger.error(f"Failed to run agent: {e}")
                self.logger.error(traceback.format_exc())
            finally:
                env.close()
                    
            # save_predictions(self.output_dir, instance.problem_statement.id, result)
            # self._chooks.on_instance_completed(result=result)
            
        except Exception as e:
            self.logger.error(f"Failed to process trajectory file {traj_path}: {e}")
            # Fallback to minimal result
            result = AgentRunResult(
                info={"exit_status": "trajectory_processing_failed"},
                trajectory=[],
            )
        finally:
            env.close()
                
        return result

    def should_skip(self, instance: BatchInstance) -> bool | str:
        """Check if we should skip this instance.
        Returns previous exit status if the instance should be skipped.
        """
        if self._redo_existing:
            return False

        # Check if there's an existing trajectory for this instance
        log_path = self.output_dir / instance.problem_statement.id / (instance.problem_statement.id + ".traj")
        if not log_path.exists():
            return False

        content = log_path.read_text()
        if not content.strip():
            self.logger.warning("Found empty trajectory: %s. Removing.", log_path)
            log_path.unlink()
            return False

        try:
            data = json.loads(content)
            # If the trajectory has no exit status, it's incomplete and we will redo it
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Found existing trajectory with no exit status: {log_path}. Removing.")
                log_path.unlink()
                return False
        except Exception as e:
            self.logger.error(f"Failed to check existing trajectory: {log_path}: {e}. Removing.")
            # If we can't check the trajectory, we will redo it
            log_path.unlink()
            return False
        # otherwise, we will skip it
        self.logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
        return exit_status

    def docker_command(self, pred: dict, env: SWEEnv, output_dir: Path, instance: BatchInstance) -> None:
        """Execute docker commands and collect execution traces."""
        try:
            try:
                env.communicate("pip install viztracer", check="raise", timeout=60)
            except Exception as e:
                self.logger.error("Failed to install viztracer: %s", e)
                
            # install astor
            env.communicate("pip install astor", check="raise", timeout=60)
                
            # copy the code from "sweagent/injector.py" to the current directory
            injector_content = open("sweagent/injector.py").read()
            env.write_file("injector.py", injector_content)
            
            # then run the injector.py with the reproduce_bug.py and reproduce_non_bug.py
            env.communicate("python injector.py reproduce_bug.py -o reproduce_bug_injected.py", check="raise", timeout=60)
            env.communicate("python injector.py reproduce_non_bug.py -o reproduce_non_bug_injected.py", check="raise", timeout=60)

            # read the reproduce_bug.py and reproduce_non_bug.py through the communicate command
            reproduce_bug_content = env.read_file("reproduce_bug_injected.py")
            reproduce_non_bug_content = env.read_file("reproduce_non_bug_injected.py")

            reproduce_bug_content = wrap_code_with_ast(reproduce_bug_content, "traceback.txt")
            reproduce_non_bug_content = wrap_code_with_ast(reproduce_non_bug_content, "traceback_non_bug.txt")

            env.write_file("reproduce_bug_wrapper.py", reproduce_bug_content)
            env.write_file("reproduce_non_bug_wrapper.py", reproduce_non_bug_content)
            try:
                env.communicate(
                    "viztracer -o trace.json --ignore_c_function --ignore_frozen reproduce_bug_wrapper.py",
                    timeout=200,
                )
                # Copy trace.json
                if env.communicate("ls trace.json", check="ignore").strip():
                    trace_content = env.read_file("trace.json")
                    (output_dir / f"{instance.problem_statement.id}.trace.json").write_text(trace_content)
                else:
                    self.logger.warning("Bug trace.json file was not created")
                # Copy traceback.txt
                if env.communicate("ls traceback.txt", check="ignore").strip():
                    tb_content = env.read_file("traceback.txt")
                    (output_dir / f"{instance.problem_statement.id}.traceback.txt").write_text(tb_content)
                else:
                    self.logger.info("No bug traceback.txt file created (no exception)")
            except Exception as e:
                self.logger.error("Failed to execute bug tracer: %s", e)

            try:
                # use python -m trace --trace reproduce_bug_wrapper.py --outfile="trace_bug_without_viztracer.txt"
                env.communicate("python -m trace --trackcalls reproduce_bug_wrapper.py > 'trace_bug_without_viztracer.txt'", timeout=200)
                # Copy trace_bug_without_viztracer.txt
                if env.communicate("ls trace_bug_without_viztracer.txt", check="ignore").strip():
                    tb_content = env.read_file("trace_bug_without_viztracer.txt")
                    (output_dir / f"{instance.problem_statement.id}.trace_bug_without_viztracer.txt").write_text(tb_content)
                else:
                    self.logger.info("No trace_bug_without_viztracer.txt file created (no exception)")
            except Exception as e:
                self.logger.error("Failed to execute bug tracer without viztracer: %s", e)

            try:
                env.communicate(
                    "viztracer -o trace_non_bug.json --ignore_c_function --ignore_frozen reproduce_non_bug_wrapper.py",
                    timeout=200,
                )
                # Copy trace_non_bug.json
                if env.communicate("ls trace_non_bug.json", check="ignore").strip():
                    trace_content = env.read_file("trace_non_bug.json")
                    (output_dir / f"{instance.problem_statement.id}.trace_non_bug.json").write_text(trace_content)
                else:
                    self.logger.warning("Non-bug trace_non_bug.json file was not created")
                # Copy traceback_non_bug.txt
                if env.communicate("ls traceback_non_bug.txt", check="ignore").strip():
                    tb_content = env.read_file("traceback_non_bug.txt")
                    (output_dir / f"{instance.problem_statement.id}.traceback_non_bug.txt").write_text(tb_content)
                else:
                    self.logger.info("No non-bug traceback_non_bug.txt file created (no exception)")
            except Exception as e:
                self.logger.error("Failed to execute non-bug tracer: %s", e)

            try:
                # use python -m trace --trace reproduce_non_bug_wrapper.py --outfile="trace_non_bug_without_viztracer.txt"
                env.communicate("python -m trace --trackcalls reproduce_non_bug_wrapper.py > 'trace_non_bug_without_viztracer.txt'", timeout=200)
                # Copy trace_non_bug_without_viztracer.txt
                if env.communicate("ls trace_non_bug_without_viztracer.txt", check="ignore").strip():
                    tb_content = env.read_file("trace_non_bug_without_viztracer.txt")
                    (output_dir / f"{instance.problem_statement.id}.trace_non_bug_without_viztracer.txt").write_text(tb_content)
                else:
                    self.logger.info("No trace_non_bug_without_viztracer.txt file created (no exception)")
            except Exception as e:
                self.logger.error("Failed to execute non-bug tracer without viztracer: %s", e)
                
            # copy all the files ends with .traceback.txt to the output_dir
            for file in env.communicate("ls *.traceback.txt", check="ignore").strip().split("\n"):
                if file:
                    tb_content = env.read_file(file)
                    (output_dir / file).write_text(tb_content)
                    with open("log.txt", "a") as log_file:
                        log_file.write(f"Copied {file} to {output_dir}\n")
        except Exception as e:
            self.logger.error("Unexpected error in docker_command: %s", e)
            self.logger.error(traceback.format_exc())

    def _add_instance_log_file_handlers(self, instance_id: str, out_dir: Path, multi_worker: bool = False) -> None:
        filename_template = f"{instance_id}.{{level}}.log"
        for level in ["trace", "debug", "info"]:
            filter = instance_id if multi_worker else ""
            add_file_handler(
                out_dir / filename_template.format(level=level),
                filter=filter,
                level=level,
                id_=f"{instance_id}-{level}",
            )

    def _remove_instance_log_file_handlers(self, instance_id: str) -> None:
        for level in ["trace", "debug", "info"]:
            remove_file_handler(f"{instance_id}-{level}")


def run_from_config(config: RunBatchConfig):
    RunBatch.from_config(config).main()


def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (  # type: ignore
        __doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunBatchConfig)
    )
    run_from_config(BasicCLI(RunBatchConfig, help_text=help_text).get_config(args))  # type: ignore


if __name__ == "__main__":
    run_from_cli()
