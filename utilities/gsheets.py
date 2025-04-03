from config.settings import constants


def log_to_gsheets(email, prompt, df):
    # constants.GSHEET.append_table(values=[f'{email}', prompt, df.to_string()])
    print(f"Email: {email}, Prompt: {prompt}, DataFrame: {df.to_string()}")
