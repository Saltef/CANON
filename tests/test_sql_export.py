import unittest

from canon.db.sql_export import sql_literal


class SqlExportTests(unittest.TestCase):
    def test_sql_literal_escapes_single_quotes(self):
        self.assertEqual(sql_literal("Lake's theory"), "'Lake''s theory'")

    def test_sql_literal_handles_booleans(self):
        self.assertEqual(sql_literal(True), "true")
        self.assertEqual(sql_literal(False), "false")


if __name__ == "__main__":
    unittest.main()
