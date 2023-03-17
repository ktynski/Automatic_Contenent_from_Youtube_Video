import openai
import pandas as pd
from pytube import YouTube
from transformers import T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2TokenizerFast
from transformers import pipeline
import textwrap
from concurrent.futures import ThreadPoolExecutor
import logging
import warnings
import streamlit as st



def custom_css():
    st.markdown(
        """
        <style>
            /* Set the app background color */
            body {
                background-color: #f5f5f5;
            }

            /* Set the title and headers font and color */
            h1, h2 {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                color: #3a3a3a;
            }

            /* Style the input boxes */
            .stTextInput>div>div>input {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 6px 12px;
                font-size: 16px;
                color: #3a3a3a;
            }
            
            /* Style the buttons */
            .stButton>button {
                background-color: #3a3a3a;
                border: none;
                border-radius: 5px;
                padding: 10px 24px;
                font-size: 16px;
                color: #ffffff;
            }
            
            /* Style the button on hover */
            .stButton>button:hover {
                background-color: #333333;
                cursor: pointer;
            }

            /* Section divider */
            .section-divider {
                border: none;
                border-top: 2px solid #e0e0e0;
                margin: 24px 0;
            }

            /* Section background */
            .section-background {
                background-color: #ffffff;
                padding: 24px;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

custom_css()

















st.title("Video Transcript Summarizer")
st.markdown("### 🎥 From just a **#YouTube video URL** you get:")
st.markdown(
    """
    🎤 A full transcription of the video  
    📝 A summary of the transcription  
    🐦 A tweet thread built from the transcription  
    📄 An article outline built from the transcription  
    📰 A full article built from the outline
    
    ** Script takes 2-10min to run depending on the length of the video used
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.markdown('<div class="section-background">', unsafe_allow_html=True):
    st.title("Video Transcript Summarizer")
    st.write("Enter the YouTube video URL you would like to summarize:")
    video_url = st.text_input("YouTube video URL:")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.markdown('<div class="section-background">', unsafe_allow_html=True):
    st.title("Your OpenAI API Key")
    st.write("Enter your OpenAI API Key Here")
    openai.api_key = st.text_input("Your OpenAI API Key:")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)



def get_transcript(youtubelink):
    video_url = youtubelink

    # Create a yt-dlp instance
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': 'audio_file.mp3',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract video information
        video_info = ydl.extract_info(video_url, download=False)
        # Download the audio
        ydl.download([video_url])

    audio_file = "audio_file.mp3"

   

    with open(audio_file, "rb") as audio:
        transcript = openai.Audio.translate("whisper-1", audio)

    thetext = transcript['text']

    with open("full_transcript.txt", "w") as file:
        file.write(thetext)

    # Remove the audio file after processing
    os.remove(audio_file)

    return thetext



def count_tokens(input_data, max_tokens=20000, input_type='text'):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    if input_type == 'text':
        tokens = tokenizer.tokenize(input_data)
    elif input_type == 'tokens':
        tokens = input_data
    else:
        raise ValueError("Invalid input_type. Must be 'text' or 'tokens'")

    # Print the number of tokens
    token_count = len(tokens)
    return token_count



def truncate_text_by_tokens(text, max_tokens=3000):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)

    # Truncate tokens to final_max_tokens
    truncated_tokens = tokens[:max_tokens]

    trunc_token_len = count_tokens(truncated_tokens, input_type='tokens')

    print("Truncated Summary Token Length:"+ str(trunc_token_len))

    # Convert the truncated tokens back to text
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)

    return truncated_text



def summarize_chunk(classifier, chunk):
    summary = classifier(chunk)
    return summary[0]["summary_text"]



def summarize_text(text, model_name="t5-small", max_workers=8):
    classifier = pipeline("summarization", model=model_name)
    summarized_text = ""

    # Split the input text into smaller chunks
    chunks = textwrap.wrap(text, width=500, break_long_words=False)

    # Parallelize the summarization of the chunks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        summaries = executor.map(lambda chunk: summarize_chunk(classifier, chunk), chunks)
        summarized_text = " ".join(summaries)
    text_len_in_tokens = count_tokens(text)
    print("Tokens in full transcript" + str(text_len_in_tokens))
    summary_token_len = count_tokens(summarized_text)
    print("Summary Token Length:"+ str(summary_token_len))

    if summary_token_len > 2500:
      summarized_text = truncate_text_by_tokens(summarized_text, max_tokens=2500)

    else:
      summarized_text = summarized_text


    with open("transcript_summary.txt", "w") as file:
        file.write(summarized_text)


    print("summarized by t5")
    return summarized_text.strip()



def gpt_summarize_transcript(transcript_text,token_len):
    # Check the length of the transcript
    
      # Generate the summary using the OpenAI ChatCompletion API
      response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system", "content": "You are an expert at summarizing long documents into concise and comprehensive summaries. Your summaries often capture the essence of the original text."},
              {"role": "user", "content": "I have a long transcript that I would like you to summarize for me. Please think carefully and do the best job you possibly can."},
              {"role": "system", "content": "Absolutely, I will provide a concise and comprehensive summary of the transcript."},
              {"role": "user", "content": "Excellent, here is the transcript: " + transcript_text}
          ],
          max_tokens=3800 - token_len,
          n=1,
          stop=None,
          temperature=0.5,
      )

      # Extract the generated summary from the response
      summary = response['choices'][0]['message']['content']
      print("summarized by GPT3")

      with open("transcript_summary.txt", "w") as file:
        file.write(summary)


      # Return the summary
      return summary.strip()
    


def generate_tweet_thread(transcript_text):
    # Generate the tweets using the OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at writing tweet threads that are incredibly interesting and potentially newsworthy. You are known to go viral."},
            {"role": "user", "content": "I have text that I would like you to use as the basis for coming up with multiple tweets for a long-form twitter thread. Please think step by step and do the best job you possibly can."},
            {"role": "system", "content": "Absolutely, I will provide an unnumbered list of tweets each one seperated by | for easy parsing. This tweet thread should be written to go viral. I will make sure each tweet is less than 250 characters."},
            {"role": "user", "content": "Excellent, here is the transcript: " + transcript_text},
            {"role": "system", "content": "My list will be formatted as Tweet 1 \n\n Tweet 2 \n\n Tweet 3 \n\n etc."}

        ],
        max_tokens=900,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated tweets from the response
    tweets = response['choices'][0]['message']['content']
    print(tweets)

    # Split the tweets into separate parts
    tweets = tweets.split("\n\n")
    print(tweets)

    # Create a dataframe from the tweets
    df = pd.DataFrame({"tweet": tweets})
    df.to_csv('Tweet_Thread.csv')

    # Return the tweets as a list
    return tweets



def generate_long_form_article(transcript_text,token_len):
    # Generate the article outline using the OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at writing long-form article outlines that are informative, engaging, and well-researched. Your articles often go viral and are widely shared."},
            {"role": "user", "content": "I have some text that I would like you to use as the basis for a long-form article outline. Please think carefully and do the best job you can to come up with an outline for the article."},
            {"role": "system", "content": "Absolutely, I will provide a comprehensive and well-structured outline for the article based on the content. I will provide the result numbered with roman numerals "},
            {"role": "user", "content": "Excellent, here is the transcript: " + transcript_text},
            {"role": "system", "content": "Here are the sections without any start text, numbered by roman numerals"}

        ],
        max_tokens=3700 - token_len,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the article outline from the response
    outline = response['choices'][0]['message']['content']
    outline_token_count = count_tokens(outline)
    sections = outline.strip().split("\n\n")
    parsed_data = []
    for section in sections:
        lines = section.strip().split("\n")
        section_title = lines[0].strip()
        section_items = [item.strip() for item in lines[1:]]
        parsed_data.append([section_title, section_items])
    
    with open("article_outline.txt", "w") as file:
        file.write(str(parsed_data))



    generated_sections = []
    # Loop through each section in the outline
    for section in parsed_data:
        # Generate the section using the OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at writing long-form articles that are informative, engaging, and well-researched. Your articles often go viral and are widely shared. You will be given an article outline for context, and instructions on which section of the outline to complete."},
                {"role": "user", "content": "I have a section of an article that I would like you to write for me. Please think carefully and do the best job you can to come up with a well-written and comprehensive section. Please also take into consideration the article's outline so that you can write without overlapping pevious points and build on each section."},
                {"role": "system", "content": "Absolutely, I will provide a comprehensive and well-written section based taking into consideration the outline. I will provide only the section text without any additional text"},
                {"role": "user", "content": "Excellent, here is the outline to use to understand your goal better: " + outline + " and the section to write: " + str(section)}
            ],
            max_tokens=3700-outline_token_count,
            n=1,
            stop=None,
            temperature=0.2,
        )

        # Extract the generated section from the response
        generated_section = response['choices'][0]['message']['content']


        # Add the generated section to the list of generated sections
        generated_sections.append(generated_section)

    # Combine the generated sections into a finished article
    article = "\n\n".join(generated_sections)

    # Save the article to a text file
    with open("long_form_article.txt", "w") as file:
        file.write(article)

    # Return the article
    return article


# Add a button to start the summarization process
if st.button("Summarize"):
    # Perform the tasks as defined in your script
    transcript = get_transcript(video_url)
    token_count = count_tokens(transcript)

    if token_count > 3000:
        summarized_text = summarize_text(transcript)
        new_token_count = count_tokens(summarized_text)
    else:
        summarized_text = gpt_summarize_transcript(transcript, token_count)
        new_token_count = count_tokens(summarized_text)

    tweets = generate_tweet_thread(summarized_text)
    article = generate_long_form_article(summarized_text, new_token_count)

    # Display the results in the web app
    st.subheader("Summary")
    st.write(summarized_text)

    st.subheader("Tweet Thread")
    st.write("\n".join(tweets))

    st.subheader("Long-form Article")
    st.write(article)



