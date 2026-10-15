
import ast
import unittest
from pathlib import Path


class AgentSyntaxTests(unittest.TestCase):
    def test_agent_python_syntax(self):
        source = Path("agent.py").read_text(encoding="utf-8")
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
  
