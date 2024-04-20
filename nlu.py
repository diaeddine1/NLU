import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import requests
import xml.etree.ElementTree as ET
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def preprocess_question(question, language='en'):
    tokens = word_tokenize(question)
    pos_tags = pos_tag(tokens)
    return pos_tags

def identify_answer_type(question):
    patterns = {
        'Personne': r'\bQui\b|\bQuel ministre\b',
        'Organisation': r'\bQuelle compagnie\b',
        'Lieu': r'\bOù\b|\bDans quelle région\b',
        'Date': r'\bQuand\b|\bEn quelle année\b'
    }
    
    for entity, pattern in patterns.items():
        if re.search(pattern, question, re.IGNORECASE):
            return entity
    return 'Resource' 

def find_entity_in_dbpedia(keyword):
    response = requests.get(f"https://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=&QueryString={keyword}")
    if response.status_code == 200:
        results = response.json()
        if results:
            return results[0]['URI']
    return None

def find_relation_in_dbpedia(token):
    with open('relations.txt', 'r') as f:
        relations = [line.strip() for line in f.readlines()]
    
    if token in relations:
        return token
        best_match = min(relations, key=lambda x: nltk.edit_distance(token, x))
    if nltk.edit_distance(token, best_match) <= 2:
        return best_match
    
    synsets = wn.synsets(token)
    if synsets:
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() in relations:
                    return lemma.name()
    
    return None

def create_sparql_query(entity_uri, relation, answer_type):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX res: <http://dbpedia.org/resource/>
    SELECT DISTINCT ?uri WHERE {{
        res:{entity_uri.split('/')[-1]} dbo:{relation} ?uri .
    }}
    """
    return query

def evaluate_system():
    tree = ET.parse('questions.xml')
    root = tree.getroot()
    
    total_precision = 0
    total_recall = 0
    
    for question in root.findall('question'):
        for string in question.findall('string'):
            if string.attrib['lang'] == 'en':  
                question_text = string.text
                pos_tags = preprocess_question(question_text)
                answer_type = identify_answer_type(question_text)
                
                entities = [token[0] for token in pos_tags if token[1].startswith('NN')]
                entity_uris = {entity: find_entity_in_dbpedia(entity) for entity in entities}
                
                relations = [token[0] for token in pos_tags if token[1].startswith('VB')]
                relation = [find_relation_in_dbpedia(relation) for relation in relations if find_relation_in_dbpedia(relation)]
                
                if entity_uris and relation:
                    sparql_query = create_sparql_query(entity_uris[entities[0]], relation[0], answer_type)
                    
                   
                    correct_answers = [uri.text for uri in question.findall("./answers/answer/uri")]
                    response = requests.get(f"http://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query={sparql_query}&format=application%2Fsparql-results%2Bjson&CXML_redir_for_subjs=121&CXML_redir_for_hrefs=&timeout=30000&debug=on")
                    if response.status_code == 200:
                        results = response.json()
                        system_answers = [result['uri']['value'] for result in results['results']['bindings']]
                        
                        true_positives = len(set(correct_answers).intersection(set(system_answers)))
                        false_positives = len(set(system_answers) - set(correct_answers))
                        false_negatives = len(set(correct_answers) - set(system_answers))
                        
                        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
                        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
                        
                        total_precision += precision
                        total_recall += recall
                        
                        print(f"Question: {question_text}")
                        print(f"Correct Answers: {correct_answers}")
                        print(f"System Answers: {system_answers}")
                        print(f"Precision: {precision}")
                        print(f"Recall: {recall}")
                        print('-' * 50)
    
    num_questions = len(root.findall('question'))
    global_precision = total_precision / num_questions
    global_recall = total_recall / num_questions
    global_f1_score = 2 * (global_precision * global_recall) / (global_precision + global_recall) if global_precision + global_recall != 0 else 0
    
    print(f"Global Precision: {global_precision}")
    print(f"Global Recall: {global_recall}")
    print(f"Global F1-Score: {global_f1_score}")

evaluate_system()
