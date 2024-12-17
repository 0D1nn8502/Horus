from newspaper import * 
import json
from newspaper import Article 


def save_article_with_url(url, article_num, text_folder="articles/alJazeera", json_file="articles/alJazeera/articles.json"):
    """
    Downloads an article, saves the text in a file, and updates a JSON file with the URL mapping.
    
    Args:
        url (str): URL of the article to download.
        article_num (int): Article number for naming the text file.
        text_folder (str): Folder where article text files are stored.
        json_file (str): JSON file to store the URL mapping.
    """
    try:
        # Download and parse the article
        article = Article(url)
        article.download()
        article.parse()
        
        # Save the article text to a file
        text_filename = f"{text_folder}/article{article_num}.txt"
        with open(text_filename, "w", encoding="utf-8") as file:
            file.write(article.text)
        
        # Update the JSON file with the URL
        try:
            # Load existing data if the JSON file exists
            with open(json_file, "r", encoding="utf-8") as file:
                url_mapping = json.load(file)
        except FileNotFoundError:
            # Create a new dictionary if the file doesn't exist
            url_mapping = {}
        
        # Add the new entry and save the JSON file
        url_mapping[f"article{article_num}.txt"] = url
        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(url_mapping, file, indent=4)
        
        print(f"Saved article {article_num} and updated URL mapping.")
    except Exception as e:
        print(f"Failed to process URL {url}: {e}")



if __name__ == "__main__":
    # Example usage with a list of URLs
    urls = [
        "https://www.aljazeera.com/features/2024/12/7/the-palestinian-boy-who-wanted-to-be-like-ronaldo-killed-by-israel"
    ]
    
    for i, url in enumerate(urls, start=41):  
        save_article_with_url(url, i)


