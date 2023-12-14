from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

load_dotenv()


# Define functions to be called
def hello_function(_dict):
    return "Hello from the function!"


def bye_function(_dict):
    return "Bye from the function!"


tools = [
    Tool(
        name="Hello",
        func=hello_function,
        description="useful for greeting new users",
    ),
    Tool(
        name="Bye",
        func=bye_function,
        description="useful for when you need to say good bye to someone",
    ),
]

# Define the Langchain model and prompt
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke(
#     {
#         "input": "Hi I'm out of here."
#     }
# )
agent_executor.invoke(
    {
        "input": "Hi I'm new here."
    }
)



