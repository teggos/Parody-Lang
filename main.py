import os

lines = 0

environment = {}

file_path = os.path.join(os.getcwd(), "program.txt")

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

def tokenize(line):
    tokens = []
    i = 0
    while i < len(line):
        char = line[i]
        
        if char == "$":
            while i < len(line) and line[i] != '\n':
                i += 1
            continue

        # Skip whitespace
        if char.isspace():
            i += 1
            continue

        # Number
        if char.isdigit():
            num = char
            i += 1
            while i < len(line) and line[i].isdigit():
                num += line[i]
                i += 1
            tokens.append(('NUMBER', int(num)))
            continue

        # Identifier or keyword
        if char.isalpha():
            ident = char
            i += 1
            while i < len(line) and (line[i].isalnum() or line[i] == '_'):
                ident += line[i]
                i += 1
            if ident in ('var', 'log', 'if', 'else', 'while', 'True', 'False', 'repeat', 'function', 'input', 'return'):
                if ident == 'True':
                    tokens.append(('BOOLEAN', True))
                elif ident == 'False':
                    tokens.append(('BOOLEAN', False))
                else:
                    tokens.append((ident.upper(), ident))
            else:
                tokens.append(('IDENT', ident))
            continue

        # String literal
        if char == '"':
            i += 1
            string_val = ''
            while i < len(line) and line[i] != '"':
                string_val += line[i]
                i += 1
            if i == len(line):
                raise SyntaxError("Unterminated string literal")
            i += 1 
            tokens.append(('STRING', string_val))
            continue

        # Symbols
        elif char == '=':
            if i + 1 < len(line) and line[i + 1] == '=':
                tokens.append(('EQUAL_EQUAL', '=='))
                i += 1
            else:
                tokens.append(('EQUALS', '='))
                
        elif char == '+':
            tokens.append(('PLUS', '+'))
        elif char == '(':
            tokens.append(('LPAREN', '('))
        elif char == ')':
            tokens.append(('RPAREN', ')'))
        elif char == ';':
            tokens.append(('SEMICOLON', ';'))
        elif char == '-':
            tokens.append(('MINUS', '-'))
        elif char == '*':
            tokens.append(('MULTIPLY', '*'))
        elif char == '/':
            tokens.append(('DIVIDE', '/'))
        elif char == ':':
            tokens.append(('COLON', ':'))
        elif char == ',':
            tokens.append(('COMMA', ','))
        elif char == '<':
            if i + 1 < len(line) and line[i + 1] == '=':
                tokens.append(('LESS_EQUAL', '<='))
                i += 1
            else:
                tokens.append(('LESS', '<'))
        elif char == '>':
            if i + 1 < len(line) and line[i + 1] == '=':
                tokens.append(('GREATER_EQUAL', '>='))
                i += 1
            else:
                tokens.append(('GREATER', '>'))
        elif char == '!':
            if i + 1 < len(line) and line[i + 1] == '=':
                tokens.append(('NOT_EQUAL', '!='))
                i += 1
            else:
                tokens.append(('NOT', '!'))
            
        else:
            raise SyntaxError(f"Unexpected character '{char}' at position {i}")

        i += 1

    return tokens

def parse_expression(tokens, index):
    if not tokens:
        return None
    
    def parse_factor(tokens, index):
        token_type, token_value = tokens[index]
        if token_type == 'IDENT' and index + 1 < len(tokens) and tokens[index + 1][0] == 'LPAREN':
            func_name = token_value
            i = index + 2
            args = []
            current_arg = []
            paren_count = 1
            while i < len(tokens) and paren_count > 0:
                if tokens[i][0] == 'LPAREN':
                    paren_count += 1
                    current_arg.append(tokens[i])
                elif tokens[i][0] == 'RPAREN':
                    paren_count -= 1
                    if paren_count == 0:
                        if current_arg:
                            args.append(parse_expression(current_arg, 0))
                        break
                elif tokens[i][0] == 'COMMA' and paren_count == 1:
                    args.append(parse_expression(current_arg, 0))
                    current_arg = []
                else:
                    current_arg.append(tokens[i])
                i += 1
            if paren_count != 0:
                raise SyntaxError("Unmatched '(' in function call")
            return {'type': 'call', 'name': func_name, 'args': args}, i+1

        if token_type == 'NUMBER':
            return {'type': 'literal', 'value': token_value}, index + 1
        if token_type == 'STRING':
            return {'type': 'literal', 'value': token_value}, index + 1
        if token_type == 'IDENT':
            return {'type': 'variable', 'name': token_value}, index + 1

        
        if token_type == 'BOOLEAN':
            
            num_value = 1 if token_value is True else 0
            return {'type': 'literal', 'value': num_value}, index + 1

        elif token_type == 'LPAREN':
        
            node, next_index = parse_expression(tokens, index + 1)

            if tokens[next_index][0] != 'RPAREN':
                raise SyntaxError("Expected ')'")

            return node, next_index + 1

        else:
            raise SyntaxError(f"Unexpected token {token_type}")
        
    def parse_expr(tokens, index):
        left, index = parse_term(tokens, index)

        while index < len(tokens) and tokens[index][0] in ('PLUS', 'MINUS'):
            op = tokens[index][1]
            index += 1

            right, index = parse_term(tokens, index)

            left = {
            'type': 'binary_op',
            'operator': op,
            'left': left,
            'right': right
        }

        return left, index
    
    def parse_comparison(tokens, index):
        left, index = parse_expr(tokens, index)

        while index < len(tokens) and tokens[index][0] in ('LESS', 'GREATER', 'EQUAL_EQUAL', 'LESS_EQUAL', 'GREATER_EQUAL', 'NOT_EQUAL'):
            op_token = tokens[index]
            op = op_token[1]
            index += 1

            right, index = parse_expr(tokens, index)

            left = {
                'type': 'binary_op',
                'operator': op,
                'left': left,
                'right': right
            }

        return left, index
        
    def parse_term(tokens, index):
        left, index = parse_factor(tokens, index)

        while index < len(tokens) and tokens[index][0] in ('MULTIPLY', 'DIVIDE'):
            op = tokens[index][1]
            index += 1

            right, index = parse_factor(tokens, index)

            left = {
            'type': 'binary_op',
            'operator': op,
            'left': left,
            'right': right
        }

        return left, index
    
    
    node, final_index = parse_comparison(tokens, 0)
    
    if final_index < len(tokens):
        raise SyntaxError("Unexpected tokens after expression")
    return node


def group_statements(lines):
    grouped = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        
        if line.strip().endswith(':'):  # if or while or repeat statement
            header = line.strip()
            body = []
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                body.append(lines[i].lstrip())
                i += 1
            
            if i < len(lines) and lines[i].strip().startswith("else:"):
                else_header = lines[i].strip()
                else_body = []
                i += 1
                while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                    else_body.append(lines[i].lstrip())
                    i += 1
                grouped.append((header, body, else_header, else_body))
            else:
                grouped.append((header, body))
            
        else:
            grouped.append((line.strip(), []))
            i += 1
    return grouped


def parse(tokens):
    
    def parse_block(tokens):
    # Split tokens by `SEMICOLON` or custom logic for multi-line
        statements = []
        i = 0
        while i < len(tokens):
        # Grab tokens for one full statement
            stmt_tokens = []
            while i < len(tokens) and tokens[i][0] != 'SEMICOLON':
                stmt_tokens.append(tokens[i])
                i += 1
            i += 1  # Skip SEMICOLON if present

            if stmt_tokens:
                statements.append(parse(stmt_tokens))

        return { 'type': 'block', 'statements': statements }

    
    if not tokens:
        return None

    if tokens[0][0] == 'VAR':

        if tokens[1][0] != 'IDENT':
            raise SyntaxError("Expected variable name")

        if tokens[2][0] != 'EQUALS':
            raise SyntaxError("Expected '=' symbol")

        if tokens[3][0] not in ('NUMBER', 'STRING', 'IDENT', 'BOOLEAN'):
            raise SyntaxError("Expected a number, string, or identifier")

        return {'type': 'var_assign', 'name': tokens[1][1], 'value': {'type': 'literal', 'value': tokens[3][1]}}
    
    if tokens[0][0] == 'FUNCTION':
        if tokens[1][0] != 'IDENT':
            raise SyntaxError("Expected function name after 'function'")
        
        func_name = tokens[1][1]
        
        if tokens[2][0] != 'LPAREN':
            raise SyntaxError("Expected '(' after function name")
        paren_count = 1
        i = 3
        param_tokens = []
        while i < len(tokens) and paren_count > 0:
            if tokens[i][0] == 'LPAREN':
                paren_count += 1
            elif tokens[i][0] == 'RPAREN':
                paren_count -= 1
                if paren_count == 0:
                    break
            
            if paren_count > 0:
                if tokens[i][0] in ('IDENT', 'COMMA'):
                    param_tokens.append(tokens[i])
                elif tokens[i][0] not in ('COMMA', 'RPAREN'):
                    raise SyntaxError(f"Unexpected token '{tokens[i][1]}' in function parameters")
            i += 1
        if paren_count != 0:
            raise SyntaxError("Unmatched '(' in function definition")
        
        params = []
        temp = None

        for token_type, token_value in param_tokens:
            if token_type == 'IDENT':
                if temp is not None:
            
                    raise SyntaxError("Expected ',' between parameters")
                temp = token_value  
            elif token_type == 'COMMA':
                if temp is None:
                    raise SyntaxError("Unexpected comma in function parameters")
                params.append(temp)
                temp = None
            else:
                raise SyntaxError(f"Unexpected token '{token_type}' in function parameters")
        if temp is not None:
            params.append(temp)

        
        if i + 1 >= len(tokens) or tokens[i+1][0] != 'COLON':
            raise SyntaxError("Expected ':' after function parameters")
        
        return {'type': 'function', 'name': func_name, 'params': params, 'body': None}
    
    if tokens[0][0] == 'IDENT' and len(tokens) > 1 and tokens[1][0] == 'LPAREN':
        if tokens[-1][0] != 'RPAREN':
            raise SyntaxError("Expected closing ')' in function call")

        argument_tokens = tokens[2:-1]
        if not argument_tokens:
            return {'type': 'call','name': tokens[0][1], 'args': []}
        arguements = []
        current = []
        paren_count = 0
        for token in argument_tokens:
            if token[0] == 'LPAREN':
                paren_count += 1
                current.append(token)
            elif token[0] == 'RPAREN':
                paren_count -= 1
                current.append(token)
            elif token[0] == 'COMMA' and paren_count == 0:
                if current:
                    arguements.append(parse_expression(current, 0))
                    current = []
                else:
                    raise SyntaxError("Empty argument in function call")
            else:
                current.append(token)
        if current:
            arguements.append(parse_expression(current, 0))

        return {'type': 'call','name': tokens[0][1], 'args': arguements}
    
    if tokens[0][0] == 'INPUT':
        if tokens[1][0] != 'LPAREN' or tokens[-1][0] != 'RPAREN':
            raise SyntaxError("Expected closed parentheses for input statement")

        comma_index = None
        for i in range(2, len(tokens) - 1):
            if tokens[i][0] == 'COMMA':
                comma_index = i
                break
        if comma_index is None:
            raise SyntaxError("Expected ',' in input statement")
        
        if tokens[4][0] != 'IDENT':
            raise SyntaxError("Expected variable name after input expression")
        
        expr_tokens = tokens[2:comma_index]
        
        if not expr_tokens:
            raise SyntaxError("Empty expression in input statement")
        
        var = tokens[4][1]
        
        expr_node = parse_expression(expr_tokens, 0)

        return {'type': 'input', 'prompt': expr_node, 'variable': var}
    
    

    if tokens[0][0] == 'IF':
        # Check LPAREN after IF
        if tokens[1][0] != 'LPAREN':
            raise SyntaxError("Expected '(' after 'if'")

        # Find matching RPAREN for the IF condition
        paren_count = 1
        i = 2
        while i < len(tokens) and paren_count > 0:
            if tokens[i][0] == 'LPAREN':
                paren_count += 1
            elif tokens[i][0] == 'RPAREN':
                paren_count -= 1
            i += 1

        if paren_count != 0:
            raise SyntaxError("Unmatched '(' in if condition")

        cond_end = i
        condition_tokens = tokens[2:cond_end - 1]
        if not condition_tokens:
            raise SyntaxError("Empty condition in if statement")

        condition_node = parse_expression(condition_tokens, 0)

        # After RPAREN, expect COLON token
        if cond_end >= len(tokens) or tokens[cond_end][0] != 'COLON':
            raise SyntaxError("Expected ':' after if condition")


        return {'type': 'if', 'condition': condition_node, 'statement': None}
    
    if tokens[0][0] == 'ELSE':
        if len(tokens) < 2 or tokens[1][0] != 'COLON':
            raise SyntaxError("Expected ':' after 'else'")

        # Expect a block of statements after ELSE
        return {'type': 'else', 'statement': parse_block(tokens[2:])}
    
    if tokens[0][0] == 'RETURN':
        # Handle both: return z  and  return (z)
        if len(tokens) >= 3 and tokens[1][0] == 'LPAREN' and tokens[-1][0] == 'RPAREN':
            expr_tokens = tokens[2:-1]
        else:
            expr_tokens = tokens[1:]
        if not expr_tokens:
            raise SyntaxError("Empty expression in return statement")
        expr_node = parse_expression(expr_tokens, 0)
        return {'type': 'return', 'value': expr_node}
    
    if tokens[0][0] == 'WHILE':
        if tokens[1][0] != 'LPAREN':
            raise SyntaxError("Expected '(' after 'while'")

        paren_count = 1
        i = 2
        while i < len(tokens) and paren_count > 0:
            if tokens[i][0] == 'LPAREN':
                paren_count += 1
            elif tokens[i][0] == 'RPAREN':
                paren_count -= 1
            i += 1

        if paren_count != 0:
            raise SyntaxError("Unmatched '(' in while condition")

        cond_end = i
        condition_tokens = tokens[2:cond_end - 1]
        if not condition_tokens:
            raise SyntaxError("Empty condition in while statement")

        condition_node = parse_expression(condition_tokens, 0)

        if cond_end >= len(tokens) or tokens[cond_end][0] != 'COLON':
            raise SyntaxError("Expected ':' after while condition")
        
        return {'type': 'while', 'condition': condition_node, 'statement': None}
    
    if tokens[0][0] == 'REPEAT':
        if tokens[1][0] != 'LPAREN':
            raise SyntaxError("Expected '(' after 'repeat'")
        paren_count = 1
        i = 2
        while i < len(tokens) and paren_count > 0:
            if tokens[i][0] == 'LPAREN':
                paren_count += 1
            elif tokens[i][0] == 'RPAREN':
                paren_count -= 1
            i += 1
        if paren_count != 0:
            raise SyntaxError("Unmatched '(' in repeat condition")
        repeat_end = i
        repeat_tokens = tokens[2:repeat_end - 1]
        if not repeat_tokens:
            raise SyntaxError("Empty condition in repeat statement")
        
        count_node = parse_expression(repeat_tokens, 0)
        
        if repeat_end >= len(tokens) or tokens[repeat_end][0] != 'COLON':
            raise SyntaxError("Expected ':' after repeat condition")
        
        if not isinstance(count_node, dict) or count_node['type'] != 'literal':
            raise SyntaxError("Repeat count must be a literal value")

        return {'type': 'repeat', 'count': count_node, 'statement': None}
        

    if tokens[0][0] == 'LOG':

        if tokens[1][0] != 'LPAREN' or tokens[-1][0] != 'RPAREN':
            raise SyntaxError("Expected closed parentheses")

        expr_tokens = tokens[2:-1]
        if not expr_tokens:
            raise SyntaxError("Empty expression in log statement")

        expr_node = parse_expression(expr_tokens, 0)

        return {'type': 'log', 'output': expr_node}

    if tokens[0][0] == 'IDENT' and tokens[1][0] == 'EQUALS':
        expr_node = parse_expression(tokens[2:], 0)

        return {'type': 'assignment', 'name': tokens[0][1], 'value': expr_node}

        
def interpert(node, env):
    
    if node is None:
        return None
    # Add this at the top of your function
    if isinstance(node, list):
        results = []
        for n in node:
            results.append(interpert(n, env))
        return results

    if node['type'] == 'literal':
        return node['value']

    if node['type'] == 'variable':
        return env.get(node['name'], 0)

    if node['type'] == 'var_assign':
        env[node['name']] = interpert(node['value'], env)
        return
    
    if node['type'] == 'input':
        if node['variable'] not in env:
            raise NameError(f"Variable '{node['variable']}' is not defined")
        prompt = interpert(node['prompt'], env)
        user_input = input(prompt + " ")
        
        try:
            value = int(user_input)
        except ValueError:
            value = user_input.strip('"')
        env[node['variable']] = value
        return

    if node['type'] == 'assignment':
        value = interpert(node['value'], env)
        env[node['name']] = value
        return

    if node['type'] == 'log':
        value = interpert(node['output'], env)
        print(value)
        return

    if node['type'] == 'binary_op':
        left = interpert(node['left'], env)
        right = interpert(node['right'], env)
        op = node['operator']

        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        elif op == '<':
            return int(left < right)
        elif op == '>':
            return int(left > right)
        elif op == '==':
            return int(left == right)
        elif op == '<=':
            return int(left <= right)
        elif op == '>=':
            return int(left >= right)
        elif op == '!=':
            return int(left != right)
    
    if node['type'] == 'if':
        condition = interpert(node['condition'], env)
        if int(bool(condition)):
            return interpert(node['statement'], env)
        else:
            if 'else' in node:
                return interpert(node['else'], env)
    if node['type'] == 'repeat':
        if 'count' not in node:
            raise SyntaxError("Repeat statement requires a count")
        count = interpert(node['count'], env)
        for i in range(count):
            interpert(node['statement'], env)
    if node['type'] == 'while':
        while True:
            condition = interpert(node['condition'], env)
            if not int(bool(condition)):
                break
            interpert(node['statement'], env)
            
    if node['type'] == 'block':
        for stmt in node['statements']:
            interpert(stmt, env)
    if node['type'] == 'else':
        for stmt in node['statement']['statements']:
            interpert(stmt, env)
    if node['type'] == 'function':
        env[node['name']] = node
        return
    if node['type'] == 'return':
        value = interpert(node['value'], env)
        raise ReturnException(value)
    if node['type'] == 'call':
        func_name = node['name']
        if func_name not in env:
            raise NameError(f"Function '{func_name}' is not defined")
        
        func_node = env[func_name]
        if func_node['type'] != 'function':
            raise TypeError(f"'{func_name}' is not a function")
        
        
        arguement_vals = [interpert(arg, env) for arg in node['args']]
        
        local_env = env.copy()
        
        if len(func_node['params']) != len(arguement_vals):
            raise TypeError(f"Function '{func_name}' expects {len(func_node['params'])} arguments, got {len(arguement_vals)}")
        
        for param, arg in zip(func_node['params'], arguement_vals):
            if isinstance(param, list):
                if len(param) != 1:
                    raise SyntaxError(f"Invalid parameter format: {param}")
                local_env[param[0]] = arg
            else:
                local_env[param] = arg

        try:
            interpert(func_node['body'], local_env)
        except ReturnException as ret:
            return ret.value
    return None  # If no return statement is encountered



with open(file_path, "r") as file:
    source_code = file.readlines()

statements = group_statements(source_code)

for group in statements:
    
    if len(group) == 4:
        header, body_lines, else_header, else_body_lines = group
    else:
        header, body_lines = group
        else_header = else_body_lines = None
        
    header_tokens = tokenize(header)
    
    #print(f"Header Tokens: {header_tokens}")
    
    ast = parse(header_tokens)
    
    #print(f"Parsed AST: {ast}")

    if ast and ast['type'] in ('if', 'while', 'repeat'):
        block_nodes = []
        for line in body_lines:
            body_tokens = tokenize(line)
            body_ast = parse(body_tokens)
            if body_ast:
                block_nodes.append(body_ast)
        ast['statement'] = {'type': 'block', 'statements': block_nodes}

        if ast['type'] == 'if' and else_body_lines:
            else_nodes = []
            for line in else_body_lines:
                else_tokens = tokenize(line)
                else_ast = parse(else_tokens)
                if else_ast:
                    else_nodes.append(else_ast)
            ast['else'] = {'type': 'block', 'statements': else_nodes}
        
        
    if ast and ast['type'] == 'function':
            block_nodes = []
            for line in body_lines:
                body_tokens = tokenize(line)
                body_ast = parse(body_tokens)
                if body_ast:
                    block_nodes.append(body_ast)
            ast['body'] = {'type': 'block', 'statements': block_nodes}
    
    elif ast and ast['type'] == 'block':
        block_nodes = []
        for line in body_lines:
            body_tokens = tokenize(line)
            body_ast = parse(body_tokens)
            if body_ast:
                block_nodes.append(body_ast)
        ast['statements'] = block_nodes


    interpert(ast, environment)
