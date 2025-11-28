import json
import html
from datetime import datetime
import os
import argparse

def escape_html(text):
    """Escape HTML special characters and preserve formatting"""
    if not text:
        return ""
    return html.escape(str(text)).replace('\n', '<br>')

def format_code_block(text):
    """Format code blocks with syntax highlighting"""
    if not text:
        return ""
    
    # Handle list content by converting to string
    if isinstance(text, list):
        # If it's a list, join the elements or convert to JSON-like format
        if all(isinstance(item, str) for item in text):
            text = '\n'.join(text)
        else:
            text = json.dumps(text, indent=2)
    
    # Convert to string if it's not already
    text = str(text)
    
    # Simple code detection - if it contains common code patterns
    code_indicators = ['def ', 'class ', 'import ', 'from ', '#!/', '```', 'function', 'var ', 'const ', 'let ']
    is_code = any(indicator in text for indicator in code_indicators)
    
    if is_code or text.strip().startswith('```'):
        return f'<pre class="code-block"><code>{escape_html(text)}</code></pre>'
    else:
        return f'<div class="text-content">{escape_html(text)}</div>'

def create_html_template():
    """Create the HTML template with CSS styling"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWE Agent Trajectory Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            overflow-x: auto;
        }
        
        .nav-tab {
            padding: 15px 25px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            color: #6c757d;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            white-space: nowrap;
        }
        
        .nav-tab:hover {
            background: #e9ecef;
            color: #495057;
        }
        
        .nav-tab.active {
            color: #007bff;
            border-bottom-color: #007bff;
            background: white;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
            margin-bottom: 20px;
        }
        
        .info-card h3 {
            color: #495057;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .info-card p {
            color: #6c757d;
            margin-bottom: 5px;
        }
        
        .step {
            background: white;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            border: 1px solid #e9ecef;
        }
        
        .step-header {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }
        
        .step-header:hover {
            background: linear-gradient(135deg, #218838 0%, #1ba085 100%);
        }
        
        .step-header h3 {
            font-size: 1.3em;
            margin: 0;
        }
        
        .step-number {
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .step-content {
            padding: 20px;
        }
        
        .step-section {
            margin-bottom: 20px;
        }
        
        .step-section h4 {
            color: #495057;
            margin-bottom: 10px;
            font-size: 1.1em;
            display: flex;
            align-items: center;
        }
        
        .step-section h4::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 20px;
            background: #007bff;
            margin-right: 10px;
            border-radius: 2px;
        }
        
        .thought-section h4::before { background: #6f42c1; }
        .action-section h4::before { background: #fd7e14; }
        .observation-section h4::before { background: #dc3545; }
        .response-section h4::before { background: #28a745; }
        
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }
        
        .text-content {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
        }
        
        .execution-time {
            background: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        .working-dir {
            background: #f3e5f5;
            color: #7b1fa2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
            font-family: monospace;
        }
        
        .history-item {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #6c757d;
        }
        
        .history-item.user {
            border-left-color: #007bff;
            background: #e3f2fd;
        }
        
        .history-item.assistant {
            border-left-color: #28a745;
            background: #e8f5e8;
        }
        
        .history-item.system {
            border-left-color: #ffc107;
            background: #fff3cd;
        }
        
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .role-badge {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .role-badge.user { background: #007bff; color: white; }
        .role-badge.assistant { background: #28a745; color: white; }
        .role-badge.system { background: #ffc107; color: #212529; }
        
        .toggle-btn {
            background: none;
            border: none;
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            padding: 5px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }
        
        .toggle-btn:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .collapsible {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .collapsible.expanded {
            max-height: none;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .nav-tab {
                padding: 10px 15px;
                font-size: 0.9em;
            }
            
            .tab-content {
                padding: 20px;
            }
            
            .info-grid {
                flex-direction: column;
            }
            
            .info-card {
                min-width: unset;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 SWE Agent Trajectory</h1>
            <p>Interactive visualization of agent execution and decision-making process</p>
        </div>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="nav-tab" onclick="showTab('trajectory')">🔄 Trajectory</button>
            <button class="nav-tab" onclick="showTab('history')">💬 History</button>
            <button class="nav-tab" onclick="showTab('stats')">📈 Statistics</button>
        </div>
        
        <div id="overview" class="tab-content active">
            {overview_content}
        </div>
        
        <div id="trajectory" class="tab-content">
            {trajectory_content}
        </div>
        
        <div id="history" class="tab-content">
            {history_content}
        </div>
        
        <div id="stats" class="tab-content">
            {stats_content}
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function toggleStep(stepId) {
            const content = document.getElementById(stepId);
            const button = content.previousElementSibling.querySelector('.toggle-btn');
            
            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                button.textContent = '▼';
            } else {
                content.classList.add('expanded');
                button.textContent = '▲';
            }
        }
        
        function toggleObservation(obsId) {
            const preview = document.getElementById(obsId + '-preview');
            const full = document.getElementById(obsId + '-full');
            const toggle = document.getElementById(obsId + '-toggle');
            
            if (full.style.display === 'none') {
                preview.style.display = 'none';
                full.style.display = 'block';
                toggle.textContent = '▲ Show Less';
            } else {
                preview.style.display = 'block';
                full.style.display = 'none';
                toggle.textContent = '▼ Show Full';
            }
        }
        
        // Initialize - expand first few steps
        document.addEventListener('DOMContentLoaded', function() {
            const steps = document.querySelectorAll('.collapsible');
            steps.forEach((step, index) => {
                    step.classList.add('expanded');
                    const button = step.previousElementSibling.querySelector('.toggle-btn');
                    if (button) button.textContent = '▲';
            });
        });
    </script>
</body>
</html>
"""

def create_overview_content(data):
    """Create the overview tab content"""
    info = data.get('info', {})
    
    content = f"""
    <div class="info-card">
        <h3>🔧 Environment Info</h3>
        <p><strong>Environment:</strong> {escape_html(data.get('environment', 'N/A'))}</p>
        <p><strong>SWE Agent Version:</strong> {escape_html(info.get('swe_agent_version', 'N/A'))}</p>
        <p><strong>SWE Rex Version:</strong> {escape_html(info.get('swe_rex_version', 'N/A'))}</p>
    </div>
    
    <div class="info-card">
        <h3>📊 Execution Stats</h3>
        <p><strong>Exit Status:</strong> {escape_html(info.get('exit_status', 'N/A'))}</p>
        <p><strong>Submission:</strong> {escape_html(info.get('submission', 'N/A'))}</p>
        <p><strong>Trajectory Steps:</strong> {len(data.get('trajectory', []))}</p>
    </div>
    
    <div class="info-card">
        <h3>📝 Files Modified</h3>
        <p><strong>30% threshold:</strong> {escape_html(info.get('edited_files30', 'N/A'))}</p>
        <p><strong>50% threshold:</strong> {escape_html(info.get('edited_files50', 'N/A'))}</p>
        <p><strong>70% threshold:</strong> {escape_html(info.get('edited_files70', 'N/A'))}</p>
    </div>
    
    <div class="info-card">
        <h3>💰 Model Statistics</h3>
        <p><strong>API Calls:</strong> {info.get('model_stats', {}).get('api_calls', 'N/A')}</p>
        <p><strong>Tokens Sent:</strong> {info.get('model_stats', {}).get('tokens_sent', 'N/A'):,}</p>
        <p><strong>Tokens Received:</strong> {info.get('model_stats', {}).get('tokens_received', 'N/A'):,}</p>
        <p><strong>Instance Cost:</strong> {info.get('model_stats', {}).get('instance_cost', 'N/A')}</p>
    </div>
    """
    
    return content

def create_trajectory_content(data):
    """Create the trajectory tab content"""
    trajectory = data.get('trajectory', [])
    
    if not trajectory:
        return "<p>No trajectory data available.</p>"
    
    content = "<div class='trajectory-steps'>"
    
    for i, step in enumerate(trajectory):
        step_id = f"step-{i}"
        
        # Extract step information
        action = step.get('action', '')
        observation = step.get('observation', '')
        response = step.get('response', '')
        thought = step.get('thought', '')
        execution_time = step.get('execution_time', 0)
        state = step.get('state', {})
        working_dir = state.get('working_dir', '') if state else ''
        
        content += f"""
        <div class="step">
            <div class="step-header" onclick="toggleStep('{step_id}')">
                <h3>Step {i + 1}</h3>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="execution-time">⏱️ {execution_time:.3f}s</span>
                    {f'<span class="working-dir">📁 {escape_html(working_dir)}</span>' if working_dir else ''}
                    <button class="toggle-btn">▼</button>
                </div>
            </div>
            <div id="{step_id}" class="step-content collapsible">
        """
        
        # Add thought section
        if thought:
            content += f"""
            <div class="step-section thought-section">
                <h4>🤔 Thought Process</h4>
                {format_code_block(thought)}
            </div>
            """
        
        # Add action section
        if action:
            content += f"""
            <div class="step-section action-section">
                <h4>⚡ Action</h4>
                {format_code_block(action)}
            </div>
            """
        
        # Add observation section (collapsible by default)
        if observation:
            obs_id = f"obs-{i}"
            # Truncate observation preview
            obs_preview = observation[:200] + "..." if len(str(observation)) > 200 else observation
            content += f"""
            <div class="step-section observation-section">
                <h4 style="cursor: pointer; display: flex; align-items: center; justify-content: space-between;" onclick="toggleObservation('{obs_id}')">
                    <span>👀 Observation</span>
                    <span class="obs-toggle" id="{obs_id}-toggle">▼ Show Full</span>
                </h4>
                <div class="obs-preview" id="{obs_id}-preview">
                    {format_code_block(obs_preview)}
                </div>
                <div class="obs-full" id="{obs_id}-full" style="display: none;">
                    {format_code_block(observation)}
                </div>
            </div>
            """
        
        # Add response section
        if response:
            content += f"""
            <div class="step-section response-section">
                <h4>💬 Response</h4>
                {format_code_block(response)}
            </div>
            """
        
        content += """
            </div>
        </div>
        """
    
    content += "</div>"
    return content

def create_history_content(data):
    """Create the history tab content"""
    history = data.get('history', [])
    
    if not history:
        return "<p>No history data available.</p>"
    
    content = "<div class='history-items'>"
    
    for i, item in enumerate(history):
        role = item.get('role', 'unknown')
        content_text = item.get('content', '')
        agent = item.get('agent', '')
        message_type = item.get('message_type', '')
        
        role_class = role.lower() if role.lower() in ['user', 'assistant', 'system'] else 'unknown'
        
        content += f"""
        <div class="history-item {role_class}">
            <div class="history-header">
                <div>
                    <span class="role-badge {role_class}">{role}</span>
                    {f'<span style="margin-left: 10px; color: #6c757d;">Agent: {escape_html(agent)}</span>' if agent else ''}
                </div>
                {f'<span style="color: #6c757d; font-size: 0.9em;">{escape_html(message_type)}</span>' if message_type else ''}
            </div>
            <div class="history-content">
                {format_code_block(content_text)}
            </div>
        </div>
        """
    
    content += "</div>"
    return content

def create_stats_content(data):
    """Create the statistics tab content"""
    trajectory = data.get('trajectory', [])
    history = data.get('history', [])
    info = data.get('info', {})
    model_stats = info.get('model_stats', {})
    
    # Calculate statistics
    total_steps = len(trajectory)
    total_history_items = len(history)
    
    # Calculate execution times
    execution_times = [step.get('execution_time', 0) for step in trajectory if step.get('execution_time')]
    total_execution_time = sum(execution_times)
    avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0
    
    # Count actions by type
    action_types = {}
    for step in trajectory:
        action = step.get('action', '')
        if action:
            # Simple action type detection
            if action.startswith('bash'):
                action_type = 'bash'
            elif action.startswith('edit'):
                action_type = 'edit'
            elif action.startswith('view'):
                action_type = 'view'
            elif action.startswith('find'):
                action_type = 'find'
            elif action.startswith('cd'):
                action_type = 'cd'
            else:
                action_type = 'other'
            action_types[action_type] = action_types.get(action_type, 0) + 1
    
    content = f"""
    <div class="info-card">
        <h3>🔧 Action Distribution</h3>
    """
    
    for action_type, count in sorted(action_types.items()):
        percentage = (count / total_steps * 100) if total_steps > 0 else 0
        content += f"<p><strong>{action_type.capitalize()}:</strong> {count} ({percentage:.1f}%)</p>"
    
    content += """
        </div>
        
        <div class="info-card">
            <h3>💰 Token Usage</h3>
    """
    
    content += f"""
            <p><strong>API Calls:</strong> {model_stats.get('api_calls', 0)}</p>
            <p><strong>Tokens Sent:</strong> {model_stats.get('tokens_sent', 0):,}</p>
            <p><strong>Tokens Received:</strong> {model_stats.get('tokens_received', 0):,}</p>
            <p><strong>Total Tokens:</strong> {model_stats.get('tokens_sent', 0) + model_stats.get('tokens_received', 0):,}</p>
        </div>
        
        <div class="info-card">
            <h3>📈 Performance Metrics</h3>
            <p><strong>Tokens per API Call:</strong> {(model_stats.get('tokens_sent', 0) + model_stats.get('tokens_received', 0)) / max(model_stats.get('api_calls', 1), 1):.1f}</p>
            <p><strong>Steps per Minute:</strong> {(total_steps / (total_execution_time / 60)) if total_execution_time > 0 else 0:.1f}</p>
            <p><strong>Avg Tokens per Step:</strong> {(model_stats.get('tokens_sent', 0) + model_stats.get('tokens_received', 0)) / max(total_steps, 1):.1f}</p>
        </div>
    </div>
    """
    
    return content

def convert_trajectory_to_html(json_file_path, output_path=None):
    """Convert trajectory JSON to HTML visualization"""
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = f"{base_name}_visualization.html"
    
    # Create HTML content
    template = create_html_template()
    
    overview_content = create_overview_content(data)
    trajectory_content = create_trajectory_content(data)
    history_content = create_history_content(data)
    stats_content = create_stats_content(data)
    
    # Fill template
    html_content = template.replace('{overview_content}', overview_content)
    html_content = html_content.replace('{trajectory_content}', trajectory_content)
    html_content = html_content.replace('{history_content}', history_content)
    html_content = html_content.replace('{stats_content}', stats_content)
    
    # Write HTML file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML visualization created successfully: {output_path}")
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return
    
    return output_path
def main():
    """Main function for command line usage"""

    traj_file = "/patch_generation_with_tests__claude-sonnet-4-20250514__t-0.80__p-1.00__c-3.00___swe_bench_lite_test/pylint-dev__pylint-7080/0/pylint-dev__pylint-7080.traj"
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    convert_trajectory_to_html(traj_file, f"visualizations/{os.path.basename(traj_file).replace('.traj', '.html')}")
    
    
if __name__ == "__main__":
    main()
