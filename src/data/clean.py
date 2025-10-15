import re

from html_to_markdown import convert_to_markdown


class CleanText:
    TABLE_PATTERN = r"<<TABLE:(.*?)/TABLE>>"
    IMAGE_PATTERN = r"<<IMAGE:(.*?)/IMAGE>>"

    @staticmethod
    def get_tables(text: str) -> list[str]:
        matches = re.findall(__class__.TABLE_PATTERN, text, re.DOTALL)
        return [
            convert_to_markdown(
                match.strip(),
                escape_asterisks=False,
                escape_misc=False,
                escape_underscores=False,
            ).strip()
            for match in matches
        ]

    @staticmethod
    def get_images(text: str) -> list[str]:
        matches = re.findall(__class__.IMAGE_PATTERN, text, re.DOTALL)
        return [match.strip() for match in matches]

    @staticmethod
    def remove_tables_and_images(text: str) -> str:
        removed_images = re.sub(__class__.IMAGE_PATTERN, "", text, flags=re.DOTALL)
        removed_tables = re.sub(__class__.TABLE_PATTERN, "", removed_images, flags=re.DOTALL)
        return removed_tables
