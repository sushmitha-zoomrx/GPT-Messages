from config.settings import constants


def log_to_gsheets(email, prompt, df):
    constants.GSHEET.append_table(values=[f'{email}', prompt, df.to_string()])
