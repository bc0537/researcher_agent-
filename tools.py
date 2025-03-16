from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


# custom tool

def save_to_txt(data:str,file_name:str="reseach_output.txt"):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text=f"--- Research Output---\nTimestamp: {timestamp}\n\n{data}"
    with open(file_name,"a",encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data sucessfully saved to {file_name}"

save_tool=Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save text to a file", 
)

# web search tool

search=DuckDuckGoSearchRun()

search_tool=Tool(
    name="search_tool",
    func=search.run,
    description="Search for information on the web",
)

# wikipedia tool

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)

wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

