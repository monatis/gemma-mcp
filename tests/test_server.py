from fastmcp import FastMCP

# Create a server instance
mcp = FastMCP(name="UserServer")


@mcp.tool
def know_user():
    """Provide information about the user such as name, etc.. Use it fto get and/or provide information about the user"""
    return {
        "name": "Yusuf",
        "location": "turkey",
        "occupation": "AI researcher",
        "company": "Altai",
    }


if __name__ == "__main__":
    mcp.run()
