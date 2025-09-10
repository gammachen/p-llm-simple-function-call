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

from print import slow_print_file

import colorama
from colorama import Fore, Style

# 初始化colorama
colorama.init()

# 配置OpenAI客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

model_name = "qwen2"

# model_name = "granite3.2-vision"

# 工具函数定义
def solve(symbols: str, equation: str):
    print(f"{Fore.CYAN}🔍 调用函数: {Fore.YELLOW}{solve.__name__}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}   符号: {Fore.WHITE}{symbols}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}   方程: {Fore.WHITE}{equation}{Style.RESET_ALL}")
    x = sp.symbols('x')
    _equation = sp.sympify(equation.split('=')[0])
    _equation = sp.Eq(_equation, 0)
    solutions = sp.solve(_equation, x)
    result = {"symbols": symbols, "equation": equation,
              "solutions": str(solutions)}
    print(f"{Fore.MAGENTA}✅ 结果: {Fore.WHITE}{result}{Style.RESET_ALL}")
    return result

def multiply(multiplicand: float, multiplier: float):
    print(f"{Fore.CYAN}🔍 调用函数: {Fore.YELLOW}{multiply.__name__}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}   被乘数: {Fore.WHITE}{multiplicand}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}   乘数: {Fore.WHITE}{multiplier}{Style.RESET_ALL}")
    result = {"value": multiplicand * multiplier}
    print(f"{Fore.MAGENTA}✅ 结果: {Fore.WHITE}{result}{Style.RESET_ALL}")
    return result

def my_health_report():
    '''
    获取我的健康报告
    '''
    print(f"{Fore.CYAN}📋 调用函数: {Fore.YELLOW}{my_health_report.__name__}{Style.RESET_ALL}")
    try:
        # 从本地文件中读取健康报告
        with open("health_report.md", "r") as f:
            report = f.read()
        print(f"{Fore.GREEN}✅ 成功获取健康报告{Style.RESET_ALL}")
        return report
    except Exception as e:
        print(f"{Fore.RED}❌ 读取健康报告失败: {str(e)}{Style.RESET_ALL}")
        return f"读取健康报告失败: {str(e)}"

def my_sport_report():
    '''
    获取我的运动报告
    '''
    print(f"{Fore.CYAN}🏃 调用函数: {Fore.YELLOW}{my_sport_report.__name__}{Style.RESET_ALL}")
    try:
        # 从本地文件中读取运动报告
        with open("sport_tracing.md", "r") as f:
            report = f.read()
        
        # 确保返回的内容格式清晰，让LLM能正确理解这是一个完整的报告
        print(f"{Fore.GREEN}✅ 成功获取运动报告{Style.RESET_ALL}")
        return f"完整运动报告如下：\n{report}"
    except Exception as e:
        print(f"{Fore.RED}❌ 读取运动报告失败: {str(e)}{Style.RESET_ALL}")
        return f"读取运动报告失败: {str(e)}"

def analyze_sport_report(report: str):
    '''
    分析运动报告
    '''
    print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📊 ANALYZE_SPORT_REPORT 函数被调用{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📝 接收到的report参数类型: {Fore.WHITE}{type(report)}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📝 接收到的report参数长度: {Fore.WHITE}{len(report)}字符{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📝 接收到的report参数内容预览(前2000个字符):")
    print(f"{Fore.WHITE}{report[:2000] if report else '空'}{Style.RESET_ALL}")
    
    try:
        # 使用思维链(CoT)提示直接调用OpenAI客户端
        messages = [
            {"role": "system", "content": "你是一个专业的运动医学专家，我需要你分析一份运动报告并给出详细建议。请按照思维链方式思考并给出详细分析。"},
            {"role": "user", "content": f"""请按照以下步骤进行思考：
            1. 首先理解运动报告的内容，包括运动类型、时长、频率等关键信息
            2. 分析报告中的重要指标，如运动频率、运动强度等
            3. 评估当前运动计划的合理性和可能的改进空间
            4. 基于以上分析，给出针对性的运动建议和健康指导
            5. 基于运动报告中的多维度的数据信息，绘制运动报告图表，如运动时间、运动距离、运动强度、运动频率等
            6. 分析图表数据，总结运动趋势和效果
            
            运动报告：{report}
            """}
        ]
        
        print(f"{Fore.CYAN}🚀 发送请求到OpenAI API...{Style.RESET_ALL}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.9
        )
        
        print(f"{Fore.GREEN}✅ ANALYZE_SPORT_REPORT 分析完成{Style.RESET_ALL}")
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"{Fore.RED}❌ ANALYZE_SPORT_REPORT 函数执行出错:")
        print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
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
            description="获取用户的运动报告，不需要输入参数。当用户要求分析运动报告时，**必须首先调用此工具**获取报告文本。"
        ),
        StructuredTool.from_function(
            func=analyze_sport_report,
            name="analyze_sport_report",
            description="**这是唯一能分析运动报告的工具**，当用户要求分析运动报告时，**必须使用此工具**。**必须**将通过my_sport_report工具获取的完整运动报告文本作为report参数输入。**不调用此工具，你的任务就没有完成**。"
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
        temperature=0.1  # 进一步降低温度，使模型更严格遵守指令
    )
    
    # 创建工具列表
    tools = create_langchain_tools()
    
    # 创建提示模板 - 进一步增强版本，指令更加明确和强制
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个**必须严格遵守指令**的智能助手。\n"
                    "当用户要求分析运动报告时，你**绝对不能有任何偏差**，**必须**严格按照以下步骤执行：\n"
                    "步骤1：**立即调用**`my_sport_report`工具获取完整的运动报告文本。\n"
                    "步骤2：**等待**并接收`my_sport_report`工具返回的运动报告文本。\n"
                    "步骤3：**获取报告后，你必须立即将完整的运动报告文本**作为`report`参数的值，**完整复制粘贴**到`analyze_sport_report`工具中。\n"
                    "步骤4：**等待**并接收`analyze_sport_report`工具返回的分析结果。\n"
                    "步骤5：**将`analyze_sport_report`工具返回的分析结果直接**作为最终答案提供给用户。\n"
                    "\n"
                    "**最重要的规则**：\n"
                    "- 你**绝对不能自己分析报告内容**，必须完全依赖`analyze_sport_report`工具的分析结果。\n"
                    "- 你**必须确保**执行完所有步骤，**特别是步骤3**，**不能在获取报告后就停止**。\n"
                    "- 如果你没有调用`analyze_sport_report`工具，那么你的回答就是**完全错误的**。\n"
                    "- 请在思考过程中**明确声明你将严格执行这些步骤**，并**在获取报告后立即调用analyze_sport_report工具**。\n"
                    "- 请在执行每一步时**明确告诉用户你正在执行哪一步**。"),
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
    
    # 创建Agent执行器 - 进一步优化配置
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # 进一步增加最大迭代次数
        early_stopping_method="force",
        return_intermediate_steps=True
    )
    
    return agent_executor

# 使用LangChain Agent处理请求
def process_with_langchain(message: str) -> str:
    # 创建Agent执行器
    agent_executor = create_agent_executor()
    
    try:
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎯 开始处理请求{Style.RESET_ALL}")
        print(f"{Fore.GREEN}用户请求: {Fore.YELLOW}{message}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        # 执行Agent - 设置为更直接的链式调用
        result = agent_executor.invoke({"input": message})
        
        # 专门检查是否已经调用了analyze_sport_report工具
        analyze_called = False
        final_analysis = ""
        
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🔍 Agent执行结果详情{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        
        # 打印中间步骤，用于调试
        if "intermediate_steps" in result:
            for i, step in enumerate(result["intermediate_steps"]):
                print(f"{Fore.CYAN}\n📌 步骤{i+1}:{Style.RESET_ALL}")
                if len(step) > 0 and hasattr(step[0], 'tool'):
                    print(f"{Fore.YELLOW}   调用工具: {Fore.WHITE}{step[0].tool}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}   工具参数: {Fore.WHITE}{step[0].tool_input}{Style.RESET_ALL}")
                    # 检查是否已调用analyze_sport_report
                    if step[0].tool == "analyze_sport_report":
                        analyze_called = True
                if len(step) > 1:
                    print(f"{Fore.MAGENTA}   工具返回: {Fore.WHITE}{str(step[1])[:200]}...{Style.RESET_ALL}")  # 限制输出长度
                    # 保存分析报告结果
                    if i == len(result["intermediate_steps"]) - 1:
                        final_analysis = str(step[1])
        
        # 如果没有调用analyze_sport_report，手动调用它
        if not analyze_called:
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.RED}⚠️  检测到未调用analyze_sport_report工具，正在手动调用{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
            
            # 从中间步骤中查找my_sport_report的结果
            sport_report = None
            for step in result["intermediate_steps"]:
                if len(step) > 0 and hasattr(step[0], 'tool') and step[0].tool == "my_sport_report":
                    sport_report = step[1]
                    break
            
            # 如果找到了运动报告，调用analyze_sport_report
            if sport_report:
                print(f"{Fore.GREEN}   📊 找到运动报告，正在调用analyze_sport_report进行分析...{Style.RESET_ALL}")
                final_analysis = analyze_sport_report(sport_report)
                result["output"] = final_analysis
            else:
                print(f"{Fore.RED}   ❌ 未找到运动报告，无法进行分析{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📋 处理结果{Style.RESET_ALL}")
        print(f"{Fore.GREEN}最终输出: {Fore.WHITE}{result['output']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        return result["output"]
    except Exception as e:
        print(f"{Fore.RED}❌ 处理请求时出错: {str(e)}{Style.RESET_ALL}")
        return f"处理请求时出错: {str(e)}"
    finally:
        print(f"{Fore.GREEN}✅ 处理完成{Style.RESET_ALL}")

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
            # 修复：确保正确解析参数并处理可能的空值
            try:
                print(f"{Fore.CYAN}🔍 解析analyze_sport_report参数:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}   args: {Fore.WHITE}{args}{Style.RESET_ALL}")
                args_dict = json.loads(args)
                report = args_dict.get('report')
                if report is None:
                    print(f"{Fore.RED}⚠️  警告: report参数为空，尝试获取最近的运动报告{Style.RESET_ALL}")
                    report = my_sport_report()  # 备用方案
                function_result = analyze_sport_report(report)
            except Exception as e:
                print(f"{Fore.RED}❌ 解析参数或调用analyze_sport_report时出错: {str(e)}{Style.RESET_ALL}")
                function_result = {"error": str(e)}
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

def llm_call(message):
    messages = []
    messages.append({"role": "system", "content":
        "你是一个智能助手，可以使用工具来回答用户问题。如果你需要分析运动报告，必须严格按照以下步骤操作：\n" \
        "1. 首先调用my_sport_report工具获取完整的运动报告文本\n" \
        "2. 然后将获取到的完整运动报告文本作为report参数，传递给analyze_sport_report工具进行分析\n" \
        "3. 最后将analyze_sport_report工具返回的分析结果作为最终答案"})
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2
    )
    messages.append(response.choices[0].message.model_dump())
    
    return parse_llm_response(response, messages)


# 主函数
if __name__ == "__main__":
    # # 以下是直接调用实现
    # try:
    #     # 1. 获取运动报告
    #     print("\n1. 获取运动报告...")
    #     sport_report = my_sport_report()
    #     print(f"获取到的运动报告长度: {len(sport_report)}字符")
    #     print(f"运动报告前100个字符: {sport_report[:2000] if sport_report else '空'}\n")
    #     
    #     # 2. 分析运动报告
    #     print("2. 分析运动报告...")
    #     analysis_result = analyze_sport_report(sport_report)
    #     
    #     # 3. 输出结果
    #     print("\n3. 输出分析结果:")
    #     print(analysis_result)
    #     
    # except Exception as e:
    #     print(f"处理过程中出错: {str(e)}")
    #  
    # print("\n=======处理完成======="))

    # 以下是LangChain实现，暂时注释掉
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}🚀 使用LangChain分析个人近期的运动报告{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
    result = process_with_langchain("分析我的运动报告")
    print(result)
    
    # 使用原始的OpenAI函数调用实现
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🤖 使用原始OpenAI函数调用分析个人近期的运动报告{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    try:
        result = llm_call("分析我的运动报告")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 分析结果{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{result}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ 处理过程中出错: {str(e)}{Style.RESET_ALL}")
    
    # # 以下是其他调用示例，保留以作比较
    # print("============求解方程============")
    # print(llm_call("求方程 x**2 - 4 = 0 的解"))
    # 
    # print("============大数相乘============")
    # print(llm_call("计算2024乘2025的积"))
    # 
    # print("============其他问题============")
    # print(llm_call("介绍一下sympy库的功能"))
    # 
    # print("============传统方法============")
    # print(llm_call("计算2024乘2025的积，注意，" +
    #                "请不要使用大模型的function-calling"))

    # print("===========获取个人健康报告============")
    # print(llm_call("我的健康报告"))
    # 
    # print("===========获取个人近期的运动报告============")
    # print(llm_call("近期的运动报告"))
    
    print(f"{Fore.YELLOW}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📋 比较两个简单Function call与LangChain的COT Function Call的实现{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*100}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}📊 简单Function call的实现{Style.RESET_ALL}")
    slow_print_file("llm_functioncalling_simple.py")
    
    import time
    time.sleep(2)
    
    print(f"{Fore.BLUE}🔄 LangChain的COT Function Call的实现{Style.RESET_ALL}")
    slow_print_file("llm_functioncalling_cot.py")
