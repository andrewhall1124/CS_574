import wikipedia
from transformers import pipeline
import torch
from typing import Dict, List

class WikipediaQAAgent:    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        print(f"Loading QA model: {model_name}")
        device = 0 if torch.mps.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=device
        )
        print("Model loaded successfully!\n")
        
    def search_wikipedia(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        results = []
        try:
            search_results = wikipedia.search(query, results=num_results)
            
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append({
                        'title': page.title,
                        'content': page.content[:3000]  # First 3000 chars
                    })
                except (wikipedia.DisambiguationError, wikipedia.PageError):
                    continue
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            
        return results
    
    def answer_question(self, question: str, max_context_length: int = 2000) -> Dict:
        wiki_results = self.search_wikipedia(question)
        
        if not wiki_results:
            return {
                'answer': "I couldn't find relevant information on Wikipedia.",
                'score': 0.0,
                'source': None
            }
        
        best_answer = None
        best_score = 0.0
        best_source = None
        
        # Try each Wikipedia article
        for result in wiki_results:
            context = result['content'][:max_context_length]
            
            try:
                qa_result = self.qa_pipeline(question=question, context=context)
                
                if qa_result['score'] > best_score:
                    best_answer = qa_result['answer']
                    best_score = qa_result['score']
                    best_source = result['title']
                    
            except Exception as e:
                print(f"Error processing {result['title']}: {e}")
                continue
        
        return {
            'answer': best_answer or "Could not find an answer.",
            'score': best_score,
            'source': best_source
        }


class QABenchmarkEvaluator:    
    def __init__(self, agent: WikipediaQAAgent):
        self.agent = agent
        
    def load_triviaqa_sample(self, num_questions: int = 20) -> List[Dict]:
        sample_questions = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
            {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
            {"question": "In what year did World War II end?", "answer": "1945"},
            {"question": "What is the chemical symbol for gold?", "answer": "Au"},
            {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
            {"question": "What is the smallest country in the world?", "answer": "Vatican City"},
            {"question": "Who was the first President of the United States?", "answer": "George Washington"},
            {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
            {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
            {"question": "What is the capital of Japan?", "answer": "Tokyo"},
            {"question": "In what year did the Titanic sink?", "answer": "1912"},
            {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
            {"question": "Who wrote '1984'?", "answer": "George Orwell"},
            {"question": "What is the longest river in the world?", "answer": "Nile River"},
            {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
            {"question": "What is the currency of the United Kingdom?", "answer": "Pound Sterling"},
            {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong"},
            {"question": "What is the boiling point of water at sea level?", "answer": "100 degrees Celsius"},
            {"question": "What year did the Berlin Wall fall?", "answer": "1989"},
        ]
        return sample_questions[:num_questions]
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        if not predicted:
            return False
        pred_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()
        return truth_lower in pred_lower or pred_lower in truth_lower
    
    def run_evaluation(self, num_questions: int = 20) -> Dict:
        questions = self.load_triviaqa_sample(num_questions)
        results = []
        correct = 0
        
        print(f"Evaluating on {len(questions)} questions...")
        print("=" * 80)
        
        for i, qa_pair in enumerate(questions, 1):
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            
            print(f"\n[{i}/{len(questions)}] {question}")
            
            result = self.agent.answer_question(question)
            predicted = result['answer']
            score = result['score']
            source = result['source']
            
            print(f"Answer: {predicted}")
            print(f"Confidence: {score:.3f}")
            if source:
                print(f"Source: {source}")
            print(f"Expected: {ground_truth}")
            
            is_correct = self.evaluate_answer(predicted, ground_truth)
            if is_correct:
                correct += 1
                print("CORRECT")
            else:
                print("INCORRECT")
            
            results.append({
                "question": question,
                "predicted": predicted,
                "confidence": score,
                "source": source,
                "ground_truth": ground_truth,
                "correct": is_correct
            })
            print("-" * 80)
        
        accuracy = correct / len(questions) if questions else 0
        
        # Calculate confidence metrics
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        correct_results = [r for r in results if r['correct']]
        avg_conf_correct = sum(r['confidence'] for r in correct_results) / len(correct_results) if correct_results else 0
        incorrect_results = [r for r in results if not r['correct']]
        avg_conf_incorrect = sum(r['confidence'] for r in incorrect_results) / len(incorrect_results) if incorrect_results else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(questions),
            "avg_confidence": avg_confidence,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
            "results": results
        }


def main():
    print("=" * 80)
    print("Wikipedia QA Agent - Hugging Face Model")
    print("=" * 80)
    print()
    
    # Initialize agent
    agent = WikipediaQAAgent(model_name="deepset/electra-base-squad2") # deepset/roberta-base-squad2 deepset/deberta-v3-large-squad2 deepset/electra-base-squad2
    
    # Run evaluation
    evaluator = QABenchmarkEvaluator(agent)
    evaluation_results = evaluator.run_evaluation(num_questions=20)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {evaluation_results['total']}")
    print(f"Correct Answers: {evaluation_results['correct']}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"\nAverage Confidence (All): {evaluation_results['avg_confidence']:.3f}")
    print(f"Average Confidence (Correct): {evaluation_results['avg_confidence_correct']:.3f}")
    print(f"Average Confidence (Incorrect): {evaluation_results['avg_confidence_incorrect']:.3f}")
    print("=" * 80)

if __name__ == "__main__":
    main()