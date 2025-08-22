from supabase import create_client
import os

SUPABASE_URL = os.getenv("https://qaswbcayjdjrsjdmzpjv.supabase.co")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFhc3diY2F5amRqcnNqZG16cGp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3OTMzMzgsImV4cCI6MjA3MTM2OTMzOH0.Tem99PZNZDqRQ4ngLRxwDeqw4PnwKuELG8OlGmCme-M")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_prediction(filename, eeg_data, result_class, probabilities):
    try:
        supabase.table("predictions").insert({
            "filename": filename,
            "eeg_data": eeg_data,
            "result_class": result_class,
            "probabilities": probabilities
        }).execute()
    except Exception as e:
        print("Supabase logging failed:", e)
