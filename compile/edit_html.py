import argparse


def replace_code_in_html(file_path, old_block, new_block):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        if old_block in content:
            content = content.replace(old_block, new_block)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print("Code block replaced successfully.")
        else:
            print("Old code block not found.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace a specific block of code in an HTML file.")
    parser.add_argument("file_path", help="Path to the HTML file")
    args = parser.parse_args()

    file_path = args.file_path

    # Old code block to be replaced
    old_block = '''        Reveal.initialize({
            controls: true,
            progress: true,
            history: true,
            transition: "slide",
            slideNumber: "",
            plugins: [RevealNotes]
        });'''
    # New code block to be inserted
    new_block = '''        Reveal.initialize({
            controls: true,
            progress: true,
            history: true,
            transition: "none",
            slideNumber: "",
            plugins: [RevealNotes],
            width: 1920,
            height: 1400
        });'''

    replace_code_in_html(file_path, old_block, new_block)


if __name__ == "__main__":
    main()
