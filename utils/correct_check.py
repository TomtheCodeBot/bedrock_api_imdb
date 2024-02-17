def is_substring_present(substring, string):
    lowercase_substring = substring.lower()
    lowercase_string = string.lower()
    return lowercase_substring in lowercase_string

def get_string(label):
    if label:
        return 'positive'
    else:
        return 'negative'
    
def check_correct(response, label):
    return is_substring_present(get_string(label), response)