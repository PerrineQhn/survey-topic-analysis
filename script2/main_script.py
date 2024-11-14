"""Main script for survey analysis"""
import pandas as pd
from llm_module import LLMInterface
from topic_analyzer import TopicAnalyzer

def main():
    # Charger le modèle LLM
    llm = LLMInterface()
    analyzer = TopicAnalyzer(llm)
    
    # Charger les données
    file_path = input("Chemin du fichier Excel: ")
    df = pd.read_excel(file_path)
    
    # Afficher les colonnes disponibles
    print("\nColonnes disponibles:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    # Sélectionner la colonne à analyser
    col_idx = int(input("\nNuméro de la colonne à analyser: "))
    text_column = df.columns[col_idx]
    
    # Analyser les topics
    results = analyzer.tag_responses(df, text_column)
    
    # Afficher les résultats
    print("\nTopics identifiés:")
    for topic_id, info in analyzer.topics.items():
        print(f"\nTopic {topic_id}: {info['name']}")
        print(f"Keywords: {', '.join(info['keywords'])}")
        print(f"Size: {info['size']} responses")
    
    # Options d'édition
    while True:
        print("\nOptions:")
        print("1: Éditer un topic")
        print("2: Voir les métriques")
        print("3: Sauvegarder et quitter")
        
        choice = input("\nChoix: ")
        
        if choice == "1":
            topic_id = int(input("ID du topic à éditer: "))
            new_name = input("Nouveau nom (Enter pour passer): ")
            new_keywords = input("Nouveaux mots-clés (séparés par des virgules, Enter pour passer): ")
            
            analyzer.edit_topic(
                topic_id,
                new_name if new_name else None,
                new_keywords.split(",") if new_keywords else None
            )
            
        elif choice == "2":
            metrics = analyzer.get_quality_metrics()
            print("\nMétriques:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
                
        elif choice == "3":
            output_path = input("Nom du fichier de sortie (Excel): ")
            results.to_excel(output_path, index=False)
            print(f"Résultats sauvegardés dans {output_path}")
            break

if __name__ == "__main__":
    main()
