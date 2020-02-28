# Follow this tutorial to create an account for the Google API: 
#https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html

import gspread
from oauth2client.service_account import ServiceAccountCredentials


# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
type1_sheet = client.open("Error Analysis").worksheet("type1_errors")
type2_sheet = client.open("Error Analysis").worksheet("type2_errors")

# Error file paths
type1_path = '/home/hasan.ozturk/kanarya-github/kanarya/error_analysis/bertcustom_2020-01-09_02_type1_errors'
type2_path = '/home/hasan.ozturk/kanarya-github/kanarya/error_analysis/bertcustom_2020-01-09_02_type2_errors'

# The timeout is 100 seconds, it throws error if the data is large. Thus, export in small chunks.
'''
f = open(type1_path, 'r')

for idx, line in enumerate(f):
  type1_sheet.update_cell(idx+1, 1, line)

f.close()
'''

f = open(type2_path, 'r')

for idx, line in enumerate(f):
  if idx >= 108:
    type2_sheet.update_cell(idx+1, 1, line)

f.close()

