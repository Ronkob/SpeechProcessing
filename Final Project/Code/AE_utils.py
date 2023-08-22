def words_file_to_list(path):
    try:
        with open(path, 'r') as file:
            word_list = file.read().split()
        return word_list
    except FileNotFoundError:
        print(f"The file '{path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []