import requests
import json 


payload = {
    
    "articleText": """Israeli soldiers opened fire on 14-year-old Naji al-Baba and his friends as they played in the woods near Hebron.

Halhul, occupied West Bank – Like kids the world over, Naji al-Baba dreamed of becoming an international football player, “just like Ronaldo”.

But – like his name, which means “survivor” – that was not to be the fate of a boy born in the occupied West Bank.

Tall for a 14-year-old, Naji was always smiling and his family remember his kindness, calmness and helpfulness to everyone around him.

He was passionate about football – practising for hours at the sports club in Halhul, just north of Hebron.

A normal boy who loved to knock a football around with the neighbourhood children after school.

His mother, Samahar al-Zamara, remembers the moment she realised Naji had grown taller than her and how he never refused a request from a friend or loved one

“He grew up before his age,” the 40-year-old says. “When he left us, I felt that I lost a part of me that we’ll never get back.”

One month ago, Naji was killed by Israeli soldiers while he was doing the thing he loved – playing football with his friends.

November 3 – the day Naji died – didn’t seem unusual, his father Nidal Abdel Moti al-Baba, 47, tells Al Jazeera.

"""

}


payload2 = {

    "query" : "What is China's stance on Taiwan?",  
    "k" : 5
}



url = "http://127.0.0.1:8000/genquery"       

response = requests.post(url, json=payload) 

# # Send the GET request with query parameters
# response = requests.get(url, params={"query": "What are some criticisms of trump?", "k": 5})

# Check for a successful response
if response.status_code == 200:
    # Print the JSON response
    print(response.json())    
    
else:
    print(f"Error: {response.status_code}, {response.text}")