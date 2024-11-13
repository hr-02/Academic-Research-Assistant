import google.generativeai as genai
import os
GEMINI_API_KEY = "AIzaSyATTd-PXZRg3AG9IATYnicTF5hAA4T1zG8"

genai.configure(api_key=GEMINI_API_KEY)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "gemini-key"

model = genai.GenerativeModel(model_name="gemini-1.5-flash")



