import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain.messages import SystemMessage

system_message = """
    You re a viral twitter influencer grading a tweet. Generate critique and recommendations
    for the user's tweet

    Always provide detailed recommendations, including requests for length,
    virality, style etc.
"""

generation_system_message = """
    You are a twitter techie influencer assistant tasked with wiriting excellent twitter posts.
    Generate the best twitter post possible for the user's request.
    If the user provides critique, respond with a revised version of your previous attempts.
"""

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=system_message
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content= generation_system_message
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


llm = ChatAnthropic(
    model_name=os.environ["FCC_MODEL"],
    base_url=os.environ["FCC_BASE_URL"].removesuffix('/v1'),
    api_key=os.environ["FCC_API_KEY"],
    temperature=0
)

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm