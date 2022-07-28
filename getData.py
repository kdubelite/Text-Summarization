from youtube_transcript_api import YouTubeTranscriptApi
import os
import googleapiclient.discovery


def get_video(input_query):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    #DEVELOPER_KEY = "" // Enter your developer key here 

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.search().list(
        part="snippet",
        maxResults=1,
        order="relevance",
        q=input_query,
        type = "video",
        videoCaption = "closedCaption"
    )
    response = request.execute()

    return response

def get_transcript(input_query):
    outls = []
    try:
        video = get_video(input_query)
    except:
        outls.append("Invalid Input")
    video_id = video['items'][0]['id']['videoId']
    try:
        tx = YouTubeTranscriptApi.get_transcript(video_id, languages = ['en-IN', 'en'])
        for i in tx:
            outtxt = (i['text'])
            outls.append(outtxt)
    except:
        outls.append("There is no summary for this")
    return outls