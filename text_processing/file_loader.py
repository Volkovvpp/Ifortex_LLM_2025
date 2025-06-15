def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    print("file is loaded")
    return text