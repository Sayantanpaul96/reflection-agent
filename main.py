import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, MessagesState, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from chains import generate_chain, reflection_chain

#schema for the messages:
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

GENERATE_CHAIN = "GENERATE_CHAIN"
REFLECTION_CHAIN = "REFLECTION_CHAIN"
LAST = -1

graph = StateGraph(state_schema=MessageGraph)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECTION_CHAIN

def generation_node(state: MessageGraph):
    return {
        "messages": generate_chain.invoke({
            "messages": state["messages"]
        })
    }

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({
            "messages": state["messages"]
        })
    return {
        "messages": HumanMessage(
            content=res.content
        )
    }


graph.add_node(GENERATE_CHAIN, generation_node)
graph.set_entry_point(GENERATE_CHAIN)
graph.add_node(REFLECTION_CHAIN, reflection_node)

graph.add_conditional_edges(GENERATE_CHAIN, should_continue, {
    END: END,
    REFLECTION_CHAIN: REFLECTION_CHAIN
})
graph.add_edge(REFLECTION_CHAIN, GENERATE_CHAIN)

main_graph = graph.compile()

def main():
    print("Hello from reflection-agents!")
    # graph.compile().get_graph().draw_mermaid_png(output_file_path='flow.png')

    inputs = {
        "messages": [
            HumanMessage(
                content='''
    Make this tweet better:
    @SayantanPaul
    - newly Tool Calling feature is seriously underrated.

    After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.

    Made a video covering their newest blog post
                '''
            )
        ]
    }

    res = main_graph.invoke(input=inputs)

    print(f"Response: {res}")

if __name__ == "__main__":
    main()