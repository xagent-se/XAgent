import ast
import astor
import argparse

class TracebackInjector(ast.NodeTransformer):
    def __init__(self, output_file):
        self.output_file = output_file
        self.traceback_file = output_file.replace(".py", ".traceback.txt")

    def visit_Try(self, node):
        # First, recursively visit child nodes
        self.generic_visit(node)
        
        # Process each except handler
        for handler in node.handlers:
            if handler.body:
                # Create the traceback logging code
                traceback_code = [
                    # import traceback
                    ast.Import(names=[ast.alias(name='traceback', asname=None)]),
                    # with open("traceback.txt", "w") as f:
                    #     traceback.print_exc(file=f)
                    ast.With(
                        items=[
                            ast.withitem(
                                context_expr=ast.Call(
                                    func=ast.Name(id='open', ctx=ast.Load()),
                                    args=[
                                        ast.Constant(value=self.traceback_file),
                                        ast.Constant(value='w')
                                    ],
                                    keywords=[]
                                ),
                                optional_vars=ast.Name(id='f', ctx=ast.Store())
                            )
                        ],
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id='traceback', ctx=ast.Load()),
                                        attr='print_exc',
                                        ctx=ast.Load()
                                    ),
                                    args=[],
                                    keywords=[
                                        ast.keyword(
                                            arg='file',
                                            value=ast.Name(id='f', ctx=ast.Load())
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                ]
                
                # Insert traceback code at the beginning of except block
                handler.body = traceback_code + handler.body
        
        return node

def inject_traceback_logging(input_file, output_file=None):
    """
    Read a Python file, inject traceback logging into except blocks,
    and write the modified code to output file or overwrite the original.
    
    Args:
        input_file (str): Path to the input Python file
        output_file (str, optional): Path to output file. If None, overwrites input file.
    """
    try:
        # Read the original Python file
        with open(input_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the source code into an AST
        tree = ast.parse(source_code)
        
        # Transform the AST to inject traceback logging
        injector = TracebackInjector(input_file)
        modified_tree = injector.visit(tree)
        
        # Convert the modified AST back to source code
        modified_code = astor.to_source(modified_tree)
        
        # Determine output file path
        if output_file is None:
            output_file = input_file
        
        # Write the modified code
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        print(f"Successfully processed {input_file}")
        if output_file != input_file:
            print(f"Modified code written to {output_file}")
        else:
            print(f"Original file {input_file} has been updated")
            
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except SyntaxError as e:
        print(f"Error: Syntax error in {input_file}: {e}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject traceback logging into except blocks of a Python file.")
    parser.add_argument("input_file", help="Path to the input Python file")
    parser.add_argument("-o", "--output_file", help="Path to the output file (optional, overwrites input if not specified)")
    args = parser.parse_args()

    inject_traceback_logging(args.input_file, args.output_file)