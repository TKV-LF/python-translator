# Chinese Novel Translator

A web application that translates Chinese novels to Vietnamese using DeepSeek AI. This application allows users to input a Chinese novel URL and get the translated content in Vietnamese, while maintaining navigation between chapters.

## Features

- URL-based novel content fetching
- Chinese to Vietnamese translation using DeepSeek AI
- Maintains chapter navigation
- Clean and simple interface

## Setup
uvicorn app.main:app --reload
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your DeepSeek API key:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Enter the URL of a Chinese novel chapter in the input field
2. Click "Translate" to get the Vietnamese translation
3. Use the navigation links to move between chapters

## Note

Make sure you have a valid DeepSeek API key and comply with the terms of service of both the novel websites and the translation API. 