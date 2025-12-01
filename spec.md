# Objectif
- vectoriser entièrement un docx pour capturer la sémantique des idées
- faire une comparaison sémantique 1 à 1 des idées pour repérer les chevauchements
- générer un txt qui liste les 100 chevauchements les plus importants, avec un format : score, chunk_text_1, chunk_text_2

# Comment chunker ?

il y a en gros 3 types de bloc dans mon doc :
- phrase de 3 mots à 2 lignes (25 mots max) -> séparé par des sauts de lignes ou des retours à la ligne (= 70% du doc)
- paragraphe solitaire de 50-150 mots (= 10%)
- texte écrit (= enchainement de paragraphe de 50 à 150 mots) = 10%
- 2 parties qui sont écrites mais que l'on pourrait exclure (10%)

Séparateurs : Saut de ligne et retour à la ligne (indifféremment, hormis dans texte ecrit 
+ titre sur 3 niveaux

# Workflow
- parser le docx en json
- process le json pour obtenir la distinction des 3 niveaux de titre et les paragraphes
- chunker le texte
- embed les chunk avec Qwen3-Embedding-0.6B
- compute les distances avec cos similarity
- générer le rapport txt