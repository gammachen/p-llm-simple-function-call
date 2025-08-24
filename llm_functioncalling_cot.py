import json
import os
from typing import List, Dict, Any

from openai import OpenAI
import sympy as sp
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 配置OpenAI客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

model_name = "qwen2"

# 工具函数定义
def solve(symbols: str, equation: str):
    print(f"function: {solve.__name__} \nsymbols: {symbols} \nequation: {equation}")
    x = sp.symbols('x')
    _equation = sp.sympify(equation.split('=')[0])
    _equation = sp.Eq(_equation, 0)
    solutions = sp.solve(_equation, x)
    result = {"symbols": symbols, "equation": equation,
              "solutions": str(solutions)}
    return result

def multiply(multiplicand: float, multiplier: float):
    print(
        f"function: {multiply.__name__} \nmultiplicand: {multiplicand} \nmultiplier: {multiplier}")
    result = {"value": multiplicand * multiplier}
    return result

def my_health_report():
    '''
    获取我的健康报告
    '''
    print(f"function: {my_health_report.__name__}")
    try:
        # 从本地文件中读取健康报告
        with open("health_report.md", "r") as f:
            report = f.read()
        return report
    except Exception as e:
        return f"读取健康报告失败: {str(e)}"

def my_sport_report():
    '''
    获取我的运动报告
    '''
    print(f"function: {my_sport_report.__name__}")
    try:
        # 从本地文件中读取运动报告
        with open("sport_tracing.md", "r") as f:
            report = f.read()
        # 确保返回的内容格式清晰，让LLM能正确理解这是一个完整的报告
        return f"完整运动报告如下：\n{report}"
    except Exception as e:
        return f"读取运动报告失败: {str(e)}"

def analyze_sport_report(report: str):
    '''
    分析运动报告
    '''
    print(f"function: {analyze_sport_report.__name__}")
    try:
        # 首先获取健康报告
        # health_report = my_health_report()
        
        # 使用思维链(CoT)提示直接调用OpenAI客户端
        messages = [
            {"role": "system", "content": "你是一个专业的运动医学专家，我需要你分析一份运动报告并结合健康报告给出建议。请按照思维链方式思考并给出详细分析。"},
            {"role": "user", "content": f"""请按照以下步骤进行思考：
            1. 首先理解运动报告的内容，包括运动类型、时长、频率等关键信息
            2. 分析健康报告中的重要指标，如体重、BMI、血压等
            3. 将运动情况与健康状况联系起来，评估运动对健康的影响
            4. 基于以上分析，给出针对性的运动建议和健康指导
            
            运动报告：{report}
            
            """}
        ]
        # 健康报告：{health_report}
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.9
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"分析运动报告失败: {str(e)}"

# 创建LangChain工具列表
def create_langchain_tools() -> List[Tool]:
    tools = [
        StructuredTool.from_function(
            func=solve,
            name="solve",
            description="求解方程，需要符号和方程作为输入参数"
        ),
        StructuredTool.from_function(
            func=multiply,
            name="multiply",
            description="计算大数字乘积，需要被乘数和乘数作为输入参数"
        ),
        StructuredTool.from_function(
            func=my_health_report,
            name="my_health_report",
            description="获取用户的健康报告，不需要输入参数"
        ),
        StructuredTool.from_function(
            func=my_sport_report,
            name="my_sport_report",
            description="获取用户的运动报告，不需要输入参数"
        ),
        StructuredTool.from_function(
            func=analyze_sport_report,
            name="analyze_sport_report",
            description="分析用户的运动报告，需要运动报告作为输入参数"
        )
    ]
    return tools

# 创建Agent执行器
def create_agent_executor() -> AgentExecutor:
    # 初始化LangChain的ChatOpenAI
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model=model_name,
        temperature=0.9
    )
    
    # 创建工具列表
    tools = create_langchain_tools()
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，能够使用各种工具来回答用户的问题。当需要分析运动报告时，你应该先获取运动报告，然后再进行分析。同时，你可以获取健康报告来提供更全面的分析。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # 创建内存组件
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# 使用LangChain Agent处理请求
def process_with_langchain(message: str) -> str:
    # 创建Agent执行器
    agent_executor = create_agent_executor()
    
    try:
        # 执行Agent
        result = agent_executor.invoke({"input": message})
        return result["output"]
    except Exception as e:
        return f"处理请求时出错: {str(e)}"

# 原始实现，保留以作比较
def parse_llm_response(model_response, messages):
    # 循环处理，直到没有函数调用为止
    while model_response.choices[0].message.tool_calls:
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
        
        # 继续调用模型，看是否还有新的函数调用
        model_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=0.9
        )
    
    # 没有函数调用了，返回最终结果
    return model_response.choices[0].message.content

def llm_call(message):
    messages = []
    messages.append({"role": "system", "content":
        "你是一个智能助手，可以使用工具来回答用户问题。如果你需要分析运动报告，必须先获取运动报告。"})
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

# 定义原始工具列表（用于原始实现）
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

if __name__ == "__main__":
    print("===========使用LangChain分析个人近期的运动报告============")
    result = process_with_langchain("根据个人近期的运动报告进行分析")
    print(result)

    # 以下是原始实现的调用，保留以作比较
    # print("============求解方程============")
    # print(llm_call("求方程 x**2 - 4 = 0 的解"))
    
    # print("============大数相乘============")
    # print(llm_call("计算2024乘2025的积"))
    
    # print("============其他问题============")
    # print(llm_call("介绍一下sympy库的功能"))
    
    # print("============传统方法============")
    # print(llm_call("计算2024乘2025的积，注意，" +
    #                "请不要使用大模型的function-calling"))

    # print("===========获取个人健康报告============")
    # print(llm_call("我的健康报告"))
    
    # print("===========获取个人近期的运动报告============")
    # print(llm_call("近期的运动报告"))
    
    # print("===========使用原始实现分析个人近期的运动报告============")
    # print(llm_call("根据个人近期的运动报告进行分析"))