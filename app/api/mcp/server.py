from fastmcp import FastMCP

mcp = FastMCP("sehat-link")

mcp_app = mcp.http_app(path="/mcp")

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
