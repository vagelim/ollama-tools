import ollama
import json
import requests
from rich import print
import time
import os
import logging
import sys
import argparse
import readline

from sample_functions import do_math, get_current_time, get_current_weather, query_duckduckgo
from ollama_tools import  generate_function_description, use_tools

parser = argparse.ArgumentParser(description='Chatbot example')
parser.add_argument('--logging', type=str, default='INFO', help='Logging level')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=args.logging, format='%(asctime)s - %(levelname)s - %(message)s')

tools=[
        generate_function_description(get_current_weather),
        generate_function_description(get_current_time),
        generate_function_description(do_math),
        generate_function_description(query_duckduckgo),
        ]

logging.debug("Tools:")
logging.debug(json.dumps(tools, indent=4))
functions_desc = [ f["function"]["description"] for f in tools ]
print("I am a chatbot able to do run some functions.", "Functions:\n\t",  "\n\t".join(functions_desc))
print()
functions = {function["function"]["name"]: globals()[function["function"]["name"]] for function in tools }

messages = [('system', "You are an assistant with access to tools, if you do not have a tool to deal with the user's request but you think you can answer do it so, if not explain your capabilities")]
messages = []

def query_model(messages, tools):
    response = ollama.chat(
        model='llama3.1',
        messages=[ {'role': role, 'content': content} for role,content in messages ],
        tools=tools,
    )
    return response

# Configure readline
readline.parse_and_bind('tab: complete')
readline.set_history_length(1000)

# Try to read history from file
history_file = os.path.expanduser('~/.ollama_chat_history')
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass

while True:
    try:
        query = input()
    except EOFError:
        break
    if query == "quit":
        break
    if query.strip() == "":
        continue
    messages.append(("user", query))
    response = query_model(
        messages=messages,
        tools=tools,
    )
    if response['message']['content'] == "":
        tools_calls = response['message']['tool_calls']
        logging.debug(tools_calls)
        result = use_tools(tools_calls, functions)
        messages.append(("tool", result))
        response = query_model(
            messages=messages,
            tools=tools,
            )
    result = response['message']['content']
    print(result)
    messages.append(("assistant", result))

# Save readline history
readline.write_history_file(history_file)

