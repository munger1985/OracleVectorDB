from transformers import AutoTokenizer, AutoModel

llmModel = 'glm'

llmModel = 'genai'

if llmModel == 'glm':
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
    model = model.eval()

import oci

# Create a default config using DEFAULT profile in default location
# Refer to
# https://docs.cloud.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File
# for more info
config = oci.config.from_file()
# Initialize service client with default config file
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config)

prompt = '''
You are a text content comparison AI. Your task is to compare two paragraphs and determine if they are relevant to each other or not. "Relevant" means that the main subject and action in both paragraphs are the same or very similar. 

output format:
no, irrelevant or yes relevant.

Provide a brief explanation of your decision. Here are two example paragraphs:

Example 1:

Paragraph A: A cat is eating fish.
Paragraph B: A cow is eating a fish.

Response: no, irrelevant. The paragraphs are irrelevant because the main subjects (a cat and a cow) are different, even though the action (eating a fish) is similar. 

please help me evaluate below

Paragraph A: a mountain
Paragraph B: a high mountain 
'''


def genAi(prompt):
    # Service endpoint
    endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"

    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config,
                                                                                             service_endpoint=endpoint,
                                                                                             retry_strategy=oci.retry.NoneRetryStrategy(),
                                                                                             timeout=(10, 240))
    generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
    llm_inference_request = oci.generative_ai_inference.models.CohereLlmInferenceRequest()
    llm_inference_request.prompt = prompt
    llm_inference_request.max_tokens = 600
    llm_inference_request.temperature = 1
    llm_inference_request.frequency_penalty = 0
    llm_inference_request.top_p = 0.75

    generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyafhwal37hxwylnpbcncidimbwteff4xha77n5xz4m7p6a")
    generate_text_detail.inference_request = llm_inference_request
    generate_text_detail.compartment_id = compartment_id
    generate_text_response = generative_ai_inference_client.generate_text(generate_text_detail)
    # Print result
    print("**************************Generate Texts Result**************************")
    # print(generate_text_response.data)
    print(generate_text_response.data.inference_response.generated_texts[0].text)
    return generate_text_response.data.inference_response.generated_texts[0].text


def helpcheck(context, query):
    if llmModel == 'glm':

        print("================================================")

        response, history = model.chat(tokenizer,
                                       f"""你是专门做文本内容比较的AI助理。
    您的任务是评估两段文字的匹配性。请仔细阅读以下两段文字，并基于它们的主体、行为、所处环境等描述，如果句子存在动物，动物的主体一致才算匹配，
    第一段：{context}

    第二段：{query}

    如果两段文字描述基本是匹配的,就输出yes和匹配的原因 
    如果两段文字描述基本不是在说一个事情，就输出no和不匹配的原因
        """, history=[], temperature=0.01)
        print(response)
        return response
    else:

        prompt = f'''
        You are a text content comparison AI. Your task is to compare two paragraphs and determine if they are relevant to each other or not. "Relevant" means that the main subject and action in both paragraphs are very similar, ignoring details 

        output format:
        no, irrelevant or yes relevant.

        Provide a brief explanation of your decision. Here are two example paragraphs:

        Example 1:

        Paragraph A: A cat is eating fish.
        Paragraph B: A cow is eating a fish.

        Response: no, irrelevant. The paragraphs are irrelevant because the main subjects (a cat and a cow) are different animals, even though the action (eating a fish) is similar. 

        Example 2:

        Paragraph A: A cat is eating a fish.
        Paragraph B: cats are eating many fish

        Response: yes, relevant. The paragraphs are relevant because both paragraphs are talking about the same breed of animals, they are doing the same thing(eating), and they are eating the same kind of food.

        please help me evaluate below

        Paragraph A: {context}
        Paragraph B: {query}
        '''

        return genAi(prompt)


import oci

config = oci.config.from_file()

compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"

# Initialize service client with default config file
ai_language_client = oci.ai_language.AIServiceLanguageClient(config)

contextLangCode = 1


def toEng(srcText):
    # srcText = "长颈鹿和吉娃娃在打架"
    batch_detect_dominant_language_response = ai_language_client.batch_detect_dominant_language(
        batch_detect_dominant_language_details=oci.ai_language.models.BatchDetectDominantLanguageDetails(
            documents=[
                oci.ai_language.models.DominantLanguageDocument(
                    key="EXAMPLE-key-Value",
                    text=srcText)],
            should_ignore_transliteration=True,
            chars_to_consider=333,
            # endpoint_id="ocid1.test.oc1..<unique_ID>EXAMPLE-endpointId-Value",
            compartment_id=compartment_id,
        ))
    # print(batch_detect_dominant_language_response.data)

    # Get the data from response
    # print(batch_detect_dominant_language_response.data)
    langcode = batch_detect_dominant_language_response.data.documents[0].languages[0].code
    global contextLangCode
    contextLangCode = langcode
    if 'en' == langcode:
        return srcText
    # Send the request to service, some parameters are not required, see API
    # doc for more info
    batch_language_translation_response = ai_language_client.batch_language_translation(
        batch_language_translation_details=oci.ai_language.models.BatchLanguageTranslationDetails(
            documents=[
                oci.ai_language.models.TextDocument(
                    key="EXAMPLE-key-Value",
                    text=srcText,
                    language_code=langcode)],
            compartment_id=compartment_id,
            target_language_code="en"),
    )

    # Get the data from response
    # print(batch_language_translation_response.data)
    translated_text = batch_language_translation_response.data.documents[0].translated_text
    print(translated_text)
    return translated_text


def toLocale(srcText):
    if 'en' == contextLangCode:
        return srcText
    # Send the request to service, some parameters are not required, see API
    # doc for more info
    batch_language_translation_response = ai_language_client.batch_language_translation(
        batch_language_translation_details=oci.ai_language.models.BatchLanguageTranslationDetails(
            documents=[
                oci.ai_language.models.TextDocument(
                    key="EXAMPLE-key-Value",
                    text=srcText,
                    language_code="en")],
            compartment_id=compartment_id,
            target_language_code=contextLangCode),
    )

    # Get the data from response
    # print(batch_language_translation_response.data)
    translated_text = batch_language_translation_response.data.documents[0].translated_text
    print(translated_text)
    return translated_text


def toEnglish(query):
    # response, history = model.chat(tokenizer,
    #                                f"""你是翻译助手，如果下面文字是中文，翻译为英语并输出，
    #  {query}
    # 如果是英文，则原样输出
    # """, history=[], temperature=0.1)
    # print(response)

    return response


def smartIntent(query):
    if llmModel == 'glm':
        response, history = model.chat(tokenizer,
                                       f"""你是聪明的AI助理，按照我的要求和我给你的句子输出文字，
        给定的句子是
        {query}
        这个句子说的最贴近的是下面三类的哪一类：
        A.动物的动作
        B.只列出来一些动物，没有描述动物的动作
        C.周边环境
        并给出推理的原因
        """, history=[], temperature=0.1)
        print(response)
    else:
        prompt = f"""
        You are a helpful AI assistant, Please identify and explain the intent of the user's sentence.
        The response should be either of the list: A. animal action, B. animal species, C. environment

        Examples are like below:

        Example 1:

        query: a cat is eating a fish
        response: A. animal action 

        Example 2:
        query: a cat and a fish
        response: B. animal species

        Example 3:
        query: a mountain
        response: C. environment, because it does not have any animals in it, so it should be something about the environment

        now the user query is {query}, the intent you think is ?
        """
    response = genAi(prompt)
    return response


def chat_with_temperature(input_text, temperature=0.1):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

    # Generate response with the specified temperature
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        temperature=temperature,
        do_sample=True
    )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    smartIntent('a cat catches mice')
# print(1212,chat_with_temperature("you count from 1 to 6"))
