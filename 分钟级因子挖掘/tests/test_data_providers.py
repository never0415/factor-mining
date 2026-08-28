import os
import unittest
from unittest.mock import patch


class ProviderSecurityTest(unittest.TestCase):
    def test_example_environment_contains_no_real_secret(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(root, ".env.example"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("TQ_USER=\n", text)
        self.assertIn("TQ_PASS=\n", text)

    def test_provider_modules_do_not_embed_credentials(self):
        from min_gp.data_providers import tqsdk_provider
        with open(tqsdk_provider.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("q56065614", source)
        self.assertNotIn("q51005005", source)


if __name__ == "__main__":
    unittest.main()

