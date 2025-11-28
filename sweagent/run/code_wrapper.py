import ast

def wrap_code_with_ast(code_text, output_file):
    """
    Wraps Python code text with try-except and traceback logging using AST manipulation.
    
    Args:
        code_text (str): The Python code to wrap
        
    Returns:
        str: The wrapped code with try-except and traceback logging
        
    Raises:
        SyntaxError: If the input code has invalid syntax
        ImportError: If neither astor nor Python 3.9+ ast.unparse is available
    """
    
    # Parse the original code
    original_ast = ast.parse(code_text)
    
    # Create the wrapper AST structure
    wrapper_ast = ast.Module(body=[
        # import traceback
        ast.Import(names=[ast.alias(name='traceback', asname=None)]),
        
        # try-except block
        ast.Try(
            body=original_ast.body,  # Original code goes in try block
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id='Exception', ctx=ast.Load()),
                    name='e',
                    body=[
                        # with open('traceback.txt', 'w') as f:
                        ast.With(
                            items=[
                                ast.withitem(
                                    context_expr=ast.Call(
                                        func=ast.Name(id='open', ctx=ast.Load()),
                                        args=[
                                            ast.Constant(value=output_file),
                                            ast.Constant(value='w')
                                        ],
                                        keywords=[]
                                    ),
                                    optional_vars=ast.Name(id='f', ctx=ast.Store())
                                )
                            ],
                            body=[
                                # f.write(traceback.format_exc())
                                ast.Expr(
                                    value=ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Name(id='f', ctx=ast.Load()),
                                            attr='write',
                                            ctx=ast.Load()
                                        ),
                                        args=[
                                            ast.Call(
                                                func=ast.Attribute(
                                                    value=ast.Name(id='traceback', ctx=ast.Load()),
                                                    attr='format_exc',
                                                    ctx=ast.Load()
                                                ),
                                                args=[],
                                                keywords=[]
                                            )
                                        ],
                                        keywords=[]
                                    )
                                )
                            ]
                        )
                    ]
                )
            ],
            orelse=[],
            finalbody=[]
        )
    ], type_ignores=[])
    
    # Convert AST back to code
    try:
        import astor
        return astor.to_source(wrapper_ast)
    except ImportError:
        # Fallback: use ast.unparse if available (Python 3.9+)
        try:
            return ast.unparse(wrapper_ast)
        except AttributeError:
            raise ImportError("Need 'astor' package or Python 3.9+ for AST to source conversion. Install with: pip install astor")
