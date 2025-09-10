import json
from openai import OpenAI
import sympy as sp

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

model_name = "qwen2"

tools = [
    {
        "type": "function",
        "function": {
            "name": "solve",
            "description": "求解方程",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "description": "符号",
                        "type": "string"
                    },
                    "equation": {
                        "description": "方程",
                        "type": "string"
                    }
                },
                "required": ["symbols", "equation"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算大数字乘积",
            "parameters": {
                "type": "object",
                "properties": {
                    "multiplicand": {
                        "description": "被乘数",
                        "type": "float"
                    },
                    "multiplier": {
                        "description": "乘数",
                        "type": "float",
                    }
                },
                "required": ["multiplicand", "multiplier"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "my_health_report",
            "description": "获取我的健康报告",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "my_sport_report",
            "description": "获取我的运动报告",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sport_report",
            "description": "分析我的运动报告",
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {
                        "description": "运动报告",
                        "type": "string"
                    }
                },
                "required": [
                    "report"
                ]
            },
        }
    }
]


def solve(symbols: str, equation: str):
    print("function: {} \nsymbols: {} \nequation: {}".format(
        'solve', symbols, equation))
    x = sp.symbols('x')
    _equation = sp.sympify(equation.split('=')[0])
    _equation = sp.Eq(_equation, 0)
    solutions = sp.solve(_equation, x)
    result = {"symbols": symbols, "equation": equation,
              "solutions": str(solutions)}
    return result


def multiply(multiplicand: float, multiplier: float):
    print(
        "function: {} \nmultiplicand: {} \nmultiplier: {}".format(
            'multiply', multiplicand, multiplier))
    result = {"value": multiplicand * multiplier}
    return result

def my_health_report():
    '''
    get my health report pushed from health app,
    and return the report
    '''
    print("function: my_health_report")
    report = "我最近的健康报告是：..."
    # 从本地文件中读取健康报告
    with open("health_report.md", "r") as f:
        report = f.read()
        
    return report

def my_sport_report():
    '''
    get my sport report pushed from health app,
    and return the report
    '''
    print("function: my_sport_report")
    report = "我最近运动报告是：..."
    # 从本地文件中读取运动报告
    with open("sport_tracing.md", "r") as f:
        report = f.read()
    return report

def analyze_sport_report(report):
    '''
    analyze the sport report,
    and return the analysis
    '''
    print("function: analyze_sport_report")
    analysis = "运动报告分析结果是：..."
    # 使用LLM分析运动报告
    messages = [
        {"role": "system", "content": "You are a professional sports coach."},
        {"role": "user", "content": report},
    ]
    # 修复：使用client直接调用，而不是错误的llm_call
    model_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.9
    )
    analysis = model_response.choices[0].message.content
    return analysis

def parse_llm_response(model_response, messages):
    """
    parse the llm response,
    and return the result
    关键的步骤是解析LLM的响应，并调用相应的函数，返回结果
    """
    if model_response.choices[0].message.tool_calls:
        tool_call = model_response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        if tool_call.function.name == "solve":
            function_result = solve(**json.loads(args))
        elif tool_call.function.name == "multiply":
            function_result = multiply(**json.loads(args))
        elif tool_call.function.name == "my_health_report":
            function_result = my_health_report()
        elif tool_call.function.name == "my_sport_report":
            function_result = my_sport_report()
        elif tool_call.function.name == "analyze_sport_report":
            args_dict = json.loads(args)
            function_result = analyze_sport_report(args_dict['report'])
        else:
            function_result = {}
        messages.append({
            "role": "tool",
            "content": f"{json.dumps(function_result)}",
            "tool_call_id": tool_call.id
        })
        
        # 再次调用LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=0.9
        )
        return response.choices[0].message.content
    else:
        return model_response.choices[0].message.content


def llm_call(message: str):
    messages = []
    messages.append({"role": "system", "content":
        "你是一个精通数学计算的人工智能助理，" +
        "你可以调用函数来完成数学计算，" +
        "也可以调用函数来获取个人的健康报告和运动报告，" +
        "你可以调用函数来分析个人的运动报告，" +
        "你可以调用函数来获取个人的运动数据，"})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.9
    )
    messages.append(response.choices[0].message.model_dump())
    return parse_llm_response(response, messages)


if __name__ == "__main__":
    print("============求解方程============")
    print(llm_call("求方程 x**2 - 4 = 0 的解"))
    
    print("============大数相乘============")
    print(llm_call("计算2024乘2025的积"))
    
    print("============其他问题============")
    print(llm_call("介绍一下sympy库的功能"))
    
    print("============传统方法============")
    print(llm_call("计算2024乘2025的积，注意，" +
                   "请不要使用大模型的function-calling"))

    print("===========获取个人健康报告============")
    print(llm_call("我的健康报告"))
    
    print("===========获取个人近期的运动报告============")
    print(llm_call("近期的运动报告"))
    
    print("===========分析个人近期的运动报告============")
    print(llm_call("根据个人近期的运动报告进行分析"))