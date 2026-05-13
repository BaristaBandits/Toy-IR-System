from sentenceSegmentation import SentenceSegmentation

def evaluate_adversarial_suite():
    segmenter = SentenceSegmentation()
    
    # Text block using your 15 adversarial sentences [cite: 55]
    adversarial_text = (
        "Dr. Smith arrived at 10 a.m. He started the meeting immediately. "
        "The U.S. economy grew by 2.5%. Analysts were surprised. "
        "She bought 3.14 kg of apples. They were fresh. "
        "Mr. Brown lives on St. Patrick St. He moved there last year. "
        "Wait... are you serious? I can't believe it. "
        "He shouted, \"Stop!\" Then everyone froze. "
        "The meeting is at 5 p.m. Please be on time. "
        "She earned a Ph.D. in Physics. Now she teaches. "
        "The temperature rose to 37.5°C. It was very hot. "
        "I met Mrs. Taylor today. She seemed happy. "
        "He said, \"I will come tomorrow.\" Then he left. "
        "The version released was v2.0. It fixed many bugs. "
        "Wow!!! That was amazing. "
        "The company is called Alpha Inc. It was founded in 2001. "
        "She moved to Washington D.C. last year. She loves it there."
    )
    
    # Test each method implemented in your search engine [cite: 30]
    naive_count = len(segmenter.naive(adversarial_text))
    punkt_count = len(segmenter.punkt(adversarial_text))
    spacy_count = len(segmenter.spacySegmenter(adversarial_text))
    
    print(f"Naive approach count: {naive_count} (Expected: ~30 due to false splits)")
    print(f"Punkt approach count: {punkt_count}")
    print(f"spaCy approach count: {spacy_count}")

if __name__ == "__main__":
    evaluate_adversarial_suite()