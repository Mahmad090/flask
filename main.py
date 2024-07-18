import warnings
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import whisper
from openai import OpenAI
from flask import Flask, request, render_template,jsonify, send_file
#tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_models/tts-tacotron2-ljspeech")
#hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_model/tts-hifigan-ljspeech")
app = Flask(__name__)
client = OpenAI(api_key="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")
llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key="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")
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Suppose you are a call agent in a Pharmacy store and your job is to ask the following questions politely one by one to the client.
            Before asking questions say this introductary line "Hello! This is Julia from SafeHands Pharmacy. Im here to guide you through your medical scripting process." and if client inquires about you tell him the same.
            Do not say "Thank you" after every question and do not introduce yourself after every question.
            Here are the questions
            Question 1: Facility Name
            Question 2: Patient Name
            Question 3: Patient Date of Birth
            Question 4: Script status (it can be either new scrip or amendment to the previous onew)
            Question 5: Medicines name
            Question 6: Instruction on using Medicine
            You have to ask these 5 questions one by one. Start asking first question when user input "Start" and then so on.
            If client input anything other than a Facility like software house, Law agency etc then reassure the client if it is what they means.
            If client answer anything irrelavent and answer another queation instead of the one asked then ask the question again.
            If client wants to change answer of any question at any point ask again that question and resume from where you were before.
            After Asking about medicine and its instruction ask if the user want to add another medicine data and if user says that he want to add data for multiple medicine then first ask medicine name and then instruction one by one for all medicine.
            At the end when user has answer all these questions reassure the user about the collected data and ask if he want to correct anything.
            After client confirm everything is correct extract the collected data and convert the collected data into json format and return that with a flag at start the flag must be "flag:end" and then json in new line.
            Important note: Ask only one question at a time and wait for user to answer it then ask the next question and use proper punctuations.
        """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=memory
)
whispermodel = whisper.load_model("base")
def transcribe_audio(audio_file):
    result = whispermodel.transcribe(audio_file)
    return result['text']

def generate_response(query,clear_history=False):
    if(clear_history==True):
        memory.clear()
        responce=conversation({"question": query})
        return responce['text']
    else:
        responce=conversation({"question": query})
        return responce['text']

def text_to_wav(text, filename):
    response = client.audio.speech.create(
    model="tts-1",
    voice="shimmer",
    input=text)
    response.stream_to_file(filename)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/initial_response', methods=['GET'])
def initial_response():
    filename = 'initial_response.wav'
    predefined_text=generate_response("Start",True)
    text_to_wav(predefined_text, filename)
    return send_file(filename, mimetype='audio/wav')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    file = request.files['audio']
    filename = 'user_input.wav'
    file.save(filename)
    
    # Transcribe the audio
    user_text = transcribe_audio(filename)
    
    # Generate the response
    response_text = generate_response(user_text)
    
    # Convert response text to audio
    response_filename = 'response.wav'
    text_to_wav(response_text, response_filename)
    
    return send_file(response_filename, mimetype='audio/wav')

if __name__ == '__main__':
  app.run(port=5000)
