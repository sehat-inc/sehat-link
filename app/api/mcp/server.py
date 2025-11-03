from fastmcp import FastMCP

mcp = FastMCP("sehat-link")

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
