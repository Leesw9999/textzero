import openai

class LLMDataAugmentor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key="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")

    def augment(self, texts, labels, n_per_example=1):
        augmented = []
        for text, label in zip(texts, labels):
            prompt = f"다음 문장을 같은 의미로 {n_per_example}개 생성해줘: '{text}'"
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            # 응답에서 생성된 문장 추출 (예시)
            new_texts = response.choices[0].message.content.split('\n')
            for new_text in new_texts:
                if new_text.strip():
                    augmented.append((new_text.strip(), label))
        return augmented