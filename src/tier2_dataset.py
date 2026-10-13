"""Tier-2 benchmark data: 100 documents, 55 questions.

This module exists so the honest measurement path does not depend on the
quarantined script in ``retracted/``. The dataset itself was never the problem --
it is a pure function of two literal lists, and it is legitimate. What was
fabricated were the *evaluation results* claimed from it.

It is byte-for-byte the ``create_large_scale_dataset()`` function extracted from
``retracted/option3_full_scale_evaluation.py`` on 2026-10-13, with the
logging/typing scaffolding that function needed in context added back. No record
was altered.

Properties that matter for interpreting every measurement in this repository
(verified in ``results/diagnosis.json``):

* 100 documents, 55 questions.
* Only **21 of 55** gold answers (38.2%) appear anywhere in the corpus.
* Coverage by difficulty: easy 10/10, medium 10/10, hard 1/15, very_hard 0/20.
* The oracle extractive ceiling is therefore token-F1 **0.187**, not 1.0.

The benchmark cannot rank systems. Fix that before drawing conclusions from it.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("tier2_dataset")


def create_large_scale_dataset() -> Tuple[List[str], List[Dict]]:
    """Create a broader dataset for comparative evaluation."""
    
    logger.info("Creating large-scale dataset for Option 3...")
    
    # Comprehensive corpus with 100+ documents across all major domains
    corpus = [
        # AI & Machine Learning (15 documents)
        "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior, learning, reasoning, problem-solving, perception, and language understanding like humans.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve their performance on tasks through experience without being explicitly programmed for each specific task.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to learn complex patterns and representations from large amounts of data.",
        "Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, generate, and respond to human language in meaningful and useful ways.",
        "Computer vision is an interdisciplinary field of AI that enables machines to interpret, analyze, and make decisions based on visual information from digital images, videos, and real-world environments.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward through trial and error.",
        "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information through weighted connections and activation functions.",
        "Supervised learning is a machine learning paradigm where algorithms learn from labeled training data to make predictions or decisions on new, unseen data.",
        "Unsupervised learning involves training algorithms on data without labeled examples, discovering hidden patterns, structures, or relationships within the data.",
        "Transfer learning is a machine learning technique where a model trained on one task is adapted for a related task, leveraging previously learned knowledge to improve performance.",
        "Convolutional neural networks (CNNs) are specialized neural networks particularly effective for processing grid-like data such as images, using convolutional layers to detect local features.",
        "Recurrent neural networks (RNNs) are designed to work with sequential data by maintaining internal memory states, making them suitable for tasks involving time series or natural language.",
        "Transformer architecture revolutionized NLP by using attention mechanisms to process sequences in parallel, enabling more efficient training and better performance on various language tasks.",
        "Generative adversarial networks (GANs) consist of two neural networks competing against each other: a generator creating fake data and a discriminator trying to detect fake from real data.",
        "Attention mechanisms allow neural networks to focus on specific parts of input data, improving performance on tasks requiring selective focus like machine translation and question answering.",
        
        # Science & Scientific Discoveries (20 documents)
        "Albert Einstein developed the theory of relativity in the early 20th century, fundamentally revolutionizing our understanding of space, time, gravity, energy, and the structure of the universe.",
        "Charles Darwin formulated the theory of evolution by natural selection, explaining how species change and adapt over time through genetic variation, inheritance, and survival advantages.",
        "Marie Curie was a pioneering physicist and chemist who discovered the radioactive elements polonium and radium, becoming the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields.",
        "Isaac Newton formulated the three laws of motion and the law of universal gravitation, establishing the mathematical foundation of classical mechanics and revolutionizing physics and astronomy.",
        "DNA (deoxyribonucleic acid) is the hereditary material containing genetic instructions for the development, functioning, growth, reproduction, and evolution of all known living organisms and many viruses.",
        "Gregor Mendel's experiments with pea plants established the fundamental laws of inheritance, laying the foundation for the science of genetics and our understanding of hereditary traits.",
        "Louis Pasteur developed the germ theory of disease and created the first vaccines for rabies and anthrax, revolutionizing medicine and establishing the field of microbiology.",
        "Alexander Fleming discovered penicillin in 1928, the first true antibiotic, which has saved millions of lives and transformed the treatment of bacterial infections worldwide.",
        "Dmitri Mendeleev created the periodic table of elements, organizing chemical elements by their atomic properties and predicting the existence of undiscovered elements.",
        "Galileo Galilei used telescopic observations to support the heliocentric model of the solar system, fundamentally changing our understanding of Earth's place in the universe.",
        "James Watson and Francis Crick, with contributions from Rosalind Franklin, determined the double helix structure of DNA, revealing how genetic information is stored and replicated.",
        "Stephen Hawking advanced our understanding of black holes, particularly through his discovery of Hawking radiation, bridging quantum mechanics and general relativity.",
        "Nikola Tesla invented the alternating current (AC) electrical system and numerous other innovations that became fundamental to modern electrical power generation and distribution.",
        "Photosynthesis is the process by which green plants, algae, and certain bacteria convert sunlight, carbon dioxide, and water into glucose and oxygen, forming the basis of most food chains.",
        "Cellular respiration is the set of metabolic reactions that take place in cells to convert glucose and oxygen into adenosine triphosphate (ATP), carbon dioxide, and water, releasing energy for cellular activities.",
        "The water cycle describes the continuous movement of water on, above, and below Earth's surface through evaporation, condensation, precipitation, infiltration, and collection processes.",
        "Gravity is a fundamental interaction that causes any two objects with mass to attract each other, with force proportional to their masses and inversely proportional to the square of the distance between them.",
        "Evolution is the change in heritable traits of biological populations over successive generations, driven by mechanisms including natural selection, genetic drift, mutation, and gene flow.",
        "Quantum mechanics describes the physical properties of nature at the scale of atoms and subatomic particles, challenging classical physics with concepts like wave-particle duality and uncertainty principles.",
        "Climate change refers to long-term shifts in global or regional climate patterns, primarily attributed to increased atmospheric concentrations of greenhouse gases from human activities since the mid-20th century.",
        
        # Geography & World Knowledge (15 documents)
        "Paris is the capital and most populous city of France, located in north-central France on the Seine River, renowned worldwide for its culture, art, fashion, gastronomy, and historical architecture.",
        "The Eiffel Tower was constructed between 1887 and 1889 by French civil engineer Gustave Eiffel for the 1889 Exposition Universelle (World's Fair) in Paris, standing 324 meters tall including its antennas.",
        "London is the capital and largest city of England and the United Kingdom, situated on the River Thames in southeastern England, serving as a major global financial, cultural, and political center.",
        "The Great Wall of China is an ancient series of walls and fortifications stretching over 13,000 miles across northern China, built over many centuries for defense against northern invasions and raids.",
        "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas on the border between Nepal and Tibet, China, standing at 8,848.86 meters.",
        "The Amazon River is the longest river in the world, flowing approximately 4,345 miles from its source in the Peruvian Andes to its mouth at the Atlantic Ocean in Brazil.",
        "The Sahara Desert is the largest hot desert in the world, covering approximately 3.6 million square miles across North Africa, roughly the size of the United States or China.",
        "Tokyo is the capital of Japan and the world's most populous metropolitan area, located on the eastern coast of Honshu island, serving as Japan's political, economic, and cultural center.",
        "New York City consists of five boroughs (Manhattan, Brooklyn, Queens, The Bronx, and Staten Island) and is the most populous city in the United States, serving as a global hub for finance, arts, fashion, and culture.",
        "The Mediterranean Sea is a nearly enclosed sea connected to the Atlantic Ocean, bordered by Europe to the north, Africa to the south, and Asia to the east, covering approximately 965,000 square miles.",
        "Australia is both a country and continent located in Oceania, known for its unique wildlife, diverse ecosystems, and being the world's sixth-largest country by total area.",
        "Antarctica is Earth's southernmost continent, covering the South Pole and surrounded by the Southern Ocean, characterized by extreme cold, ice sheets, and unique research stations.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the planet's surface and containing more than half of the world's free water.",
        "The Himalayan mountain range spans five countries (India, Nepal, Bhutan, China, and Pakistan) and contains the world's highest peaks, including Mount Everest and K2.",
        "Russia is the largest country in the world by land area, spanning eleven time zones and extending from Eastern Europe across northern Asia to the Pacific Ocean.",
        
        # History & Culture (15 documents)
        "The Renaissance was a period of European cultural, artistic, political, and economic rebirth following the Middle Ages, spanning roughly from the 14th to the 17th century and beginning in Italy.",
        "Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath whose interests included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, literature, anatomy, geology, astronomy, botany, and paleontology. He painted the Mona Lisa and The Last Supper.",
        "Michelangelo Buonarroti (1475-1564) was an Italian Renaissance sculptor, painter, architect, and poet, considered one of the greatest artists of all time, famous for the Sistine Chapel ceiling and the statue of David.",
        "The Roman Empire was one of the largest empires in ancient history, lasting from 27 BC to 476 AD in Western Europe, controlling territory from Britain to North Africa and from Spain to the Middle East at its height.",
        "World War II (1939-1945) was the deadliest and most widespread conflict in human history, involving more than 30 countries and resulting in 50 to 80 million fatalities worldwide.",
        "The Industrial Revolution began in Britain in the late 18th century, marking the transition from hand production methods to mechanized manufacturing, fundamentally transforming society, economy, and technology.",
        "Ancient Egypt was a civilization of ancient Northeast Africa concentrated along the lower reaches of the Nile River, known for its pyramids, pharaohs, hieroglyphics, and lasting cultural contributions.",
        "The French Revolution (1789-1799) was a period of radical political and societal change in France that had a lasting impact on French history and more broadly throughout Europe.",
        "Christopher Columbus's voyages across the Atlantic Ocean initiated widespread European exploration and colonization of the Americas, fundamentally altering the course of world history.",
        "The Scientific Revolution of the 16th and 17th centuries fundamentally transformed views of society and nature, with key figures including Copernicus, Galileo, Kepler, Newton, and Descartes.",
        "World War I (1914-1918) was a global conflict primarily centered in Europe, involving many of the world's great powers and resulting in unprecedented destruction and loss of life.",
        "The Cold War was a period of geopolitical tension between the United States and Soviet Union and their respective allies from 1947 to 1991, characterized by ideological conflict and nuclear competition.",
        "The American Revolution (1775-1783) was a colonial revolt in which the thirteen American colonies gained independence from British rule, establishing the United States of America.",
        "Ancient Greece made fundamental contributions to philosophy, democracy, theater, science, and mathematics, with influential figures like Socrates, Plato, Aristotle, and Archimedes.",
        "The Byzantine Empire was the continuation of the Roman Empire in its eastern provinces, lasting for over 1,000 years until the fall of Constantinople in 1453.",
        
        # Technology & Programming (15 documents)
        "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991, known for its simplicity, readability, and versatility across multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "JavaScript is a high-level, interpreted programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS, enabling interactive web pages and dynamic user interfaces.",
        "The Internet is a global system of interconnected computer networks using standardized Internet Protocol Suite (TCP/IP) to communicate between networks and devices worldwide, revolutionizing information sharing and communication.",
        "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet, offering faster innovation and flexible resources.",
        "Blockchain technology is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography, providing transparency, immutability, and decentralization.",
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to process information in ways that could exponentially outperform classical computers for certain problems.",
        "The World Wide Web, invented by Tim Berners-Lee in 1989, is an information system where documents and resources are identified by URLs and interconnected via hypertext links.",
        "Software engineering is the systematic application of engineering approaches to the development, operation, and maintenance of software systems, emphasizing methodical processes and quality assurance.",
        "Database management systems (DBMS) are software applications that interact with users, other applications, and databases to capture and analyze data, providing efficient data storage and retrieval.",
        "Computer networks enable communication and resource sharing between computing devices, using protocols like TCP/IP to facilitate data transmission across local and wide area networks.",
        "Operating systems manage computer hardware and software resources, providing common services for computer programs and serving as intermediaries between applications and hardware.",
        "Cybersecurity involves protecting internet-connected systems including hardware, software, and data from cyberattacks, encompassing technologies, processes, and practices designed to protect networks and programs.",
        "Artificial neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections and activation functions.",
        "Version control systems track changes to files and coordinate work among multiple contributors, with Git being the most widely used distributed version control system for software development.",
        "Application programming interfaces (APIs) define methods of communication between software components, enabling different applications to interact and share data or functionality.",
        
        # Sports & Entertainment (10 documents)
        "The modern Olympic Games were revived in 1896 in Athens, Greece, by Baron Pierre de Coubertin, inspired by the ancient Olympic Games held in Olympia, Greece, from 776 BC to 393 AD.",
        "Football (soccer) is the world's most popular sport, played by over 250 million players in over 200 countries and dependencies, governed by FIFA and featuring the World Cup every four years.",
        "Basketball was invented in December 1891 by Canadian-American gym teacher Dr. James Naismith in Springfield, Massachusetts, as an indoor activity to keep students physically active during winter.",
        "Tennis evolved from the medieval French game 'jeu de paume,' with modern lawn tennis rules established in the 1870s in England by Major Walter Clopton Wingfield.",
        "The FIFA World Cup is the premier international football tournament held every four years since 1930 (except 1942 and 1946 due to World War II), considered the most prestigious tournament in international football.",
        "Baseball, often called America's pastime, evolved from older bat-and-ball games and became popular in the United States in the 19th century, spreading internationally over time.",
        "The National Basketball Association (NBA) is the premier professional basketball league in North America, featuring 30 teams and widely considered the top basketball league globally.",
        "Cricket is a bat-and-ball game particularly popular in countries that were former British colonies, with formats including Test matches, One Day Internationals, and Twenty20.",
        "Golf is a precision club-and-ball sport where players use various clubs to hit balls into holes on a course in the fewest strokes possible, with major championships including the Masters and British Open.",
        "Swimming is both a recreational activity and competitive sport involving propelling oneself through water using limbs, featured prominently in the Olympic Games with multiple disciplines and distances.",
        
        # Arts & Literature (10 documents)
        "Shakespeare, widely regarded as the greatest writer in the English language, wrote approximately 39 plays and 154 sonnets during the late 16th and early 17th centuries.",
        "Classical music encompasses a broad span of time from roughly the 9th century to the present day, with notable composers including Bach, Mozart, Beethoven, and Chopin.",
        "The Louvre Museum in Paris is the world's largest art museum and historic monument, housing thousands of works including the Mona Lisa and Venus de Milo.",
        "Photography is the art and science of creating durable images by recording light or electromagnetic radiation, revolutionizing visual documentation and artistic expression.",
        "Cinema, invented in the late 19th century, combines moving images with sound to tell stories, document reality, and create artistic expressions, becoming a dominant form of entertainment and art.",
        "Abstract art uses visual language of shape, form, color, and line to create compositions independent of visual references in the world, pioneered by artists like Kandinsky and Mondrian.",
        "Jazz music originated in the African-American communities of New Orleans in the late 19th and early 20th centuries, characterized by swing, blue notes, complex chords, and improvisation.",
        "Ballet is a highly technical form of dance with its own vocabulary based on French terminology, originating in the Italian Renaissance courts and later developed in France and Russia.",
        "Architecture is both the process and product of planning, designing, and constructing buildings and physical structures, combining functionality with aesthetic considerations.",
        "Literature encompasses written works, especially creative writing of recognized artistic value, including poetry, drama, fiction, and non-fiction across cultures and time periods."
    ]
    
    # Large-scale question set (50+ questions) with comprehensive coverage
    questions = [
        # Easy Questions - Direct Factual (10 questions)
        {'question': 'What is the capital of France?', 'answer': 'Paris', 'type': 'factual', 'difficulty': 'easy', 'domain': 'geography'},
        {'question': 'Who created the Python programming language?', 'answer': 'Guido van Rossum', 'type': 'factual', 'difficulty': 'easy', 'domain': 'technology'},
        {'question': 'What does AI stand for?', 'answer': 'Artificial intelligence', 'type': 'factual', 'difficulty': 'easy', 'domain': 'technology'},
        {'question': 'What is the tallest mountain in the world?', 'answer': 'Mount Everest', 'type': 'factual', 'difficulty': 'easy', 'domain': 'geography'},
        {'question': 'Who painted the Mona Lisa?', 'answer': 'Leonardo da Vinci', 'type': 'factual', 'difficulty': 'easy', 'domain': 'arts'},
        {'question': 'Which sport was invented by James Naismith?', 'answer': 'Basketball', 'type': 'factual', 'difficulty': 'easy', 'domain': 'sports'},
        {'question': 'What is the longest river in the world?', 'answer': 'Amazon River', 'type': 'factual', 'difficulty': 'easy', 'domain': 'geography'},
        {'question': 'Who discovered penicillin?', 'answer': 'Alexander Fleming', 'type': 'factual', 'difficulty': 'easy', 'domain': 'science'},
        {'question': 'What is the largest hot desert in the world?', 'answer': 'Sahara Desert', 'type': 'factual', 'difficulty': 'easy', 'domain': 'geography'},
        {'question': 'Who invented the World Wide Web?', 'answer': 'Tim Berners-Lee', 'type': 'factual', 'difficulty': 'easy', 'domain': 'technology'},
        
        # Medium Questions - Inference Required (10 questions)
        {'question': 'Which scientist developed the theory that revolutionized physics in the early 20th century?', 'answer': 'Albert Einstein', 'type': 'inference', 'difficulty': 'medium', 'domain': 'science'},
        {'question': 'What structure was built by Gustave Eiffel for the 1889 World\'s Fair?', 'answer': 'The Eiffel Tower', 'type': 'inference', 'difficulty': 'medium', 'domain': 'geography'},
        {'question': 'Which scientist won Nobel Prizes in two different scientific fields?', 'answer': 'Marie Curie', 'type': 'inference', 'difficulty': 'medium', 'domain': 'science'},
        {'question': 'Where were the modern Olympic Games first revived in 1896?', 'answer': 'Athens, Greece', 'type': 'inference', 'difficulty': 'medium', 'domain': 'sports'},
        {'question': 'What biological process do plants use to produce glucose and oxygen?', 'answer': 'Photosynthesis', 'type': 'inference', 'difficulty': 'medium', 'domain': 'science'},
        {'question': 'Which empire controlled territory from Britain to North Africa at its height?', 'answer': 'Roman Empire', 'type': 'inference', 'difficulty': 'medium', 'domain': 'history'},
        {'question': 'What programming language is known for its use in web development alongside HTML and CSS?', 'answer': 'JavaScript', 'type': 'inference', 'difficulty': 'medium', 'domain': 'technology'},
        {'question': 'Which artist sculpted the statue of David and painted the Sistine Chapel ceiling?', 'answer': 'Michelangelo', 'type': 'inference', 'difficulty': 'medium', 'domain': 'arts'},
        {'question': 'What global conflict lasted from 1939 to 1945?', 'answer': 'World War II', 'type': 'inference', 'difficulty': 'medium', 'domain': 'history'},
        {'question': 'Which mountain range contains the world\'s highest peaks including Mount Everest?', 'answer': 'Himalayas', 'type': 'inference', 'difficulty': 'medium', 'domain': 'geography'},
        
        # Hard Questions - Multi-hop Reasoning (15 questions)
        {'question': 'How are machine learning and artificial intelligence related in terms of their hierarchical relationship?', 'answer': 'Machine learning is a subset of artificial intelligence', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'technology'},
        {'question': 'What connects photosynthesis and cellular respiration in biological systems?', 'answer': 'They are complementary processes where photosynthesis produces glucose and oxygen that cellular respiration uses for energy', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'science'},
        {'question': 'What historical connection exists between Leonardo da Vinci and Michelangelo in the Renaissance?', 'answer': 'Both were Italian Renaissance artists who created masterpieces during the same historical period', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'arts'},
        {'question': 'How do the heights of Mount Everest and the Eiffel Tower compare, and where are they located?', 'answer': 'Mount Everest at 8,848 meters is much taller than the Eiffel Tower at 324 meters; Everest is in the Himalayas while the Eiffel Tower is in Paris', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'geography'},
        {'question': 'What theory connects Charles Darwin with species change and natural selection?', 'answer': 'Darwin developed the theory of evolution by natural selection to explain how species change over time', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'science'},
        {'question': 'How are Python and JavaScript similar and different as programming languages?', 'answer': 'Both are high-level interpreted languages, but Python is known for versatility while JavaScript is primarily for web development', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'technology'},
        {'question': 'What connects the ancient Olympic Games with the modern Olympics revived in 1896?', 'answer': 'The modern Olympics were inspired by and revived the ancient Greek Olympic tradition', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'sports'},
        {'question': 'How do Isaac Newton\'s contributions relate to classical mechanics and physics?', 'answer': 'Newton formulated the laws of motion and universal gravitation, establishing the foundation of classical mechanics', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'science'},
        {'question': 'What role did Marie Curie play in advancing both physics and chemistry?', 'answer': 'She discovered radioactive elements and won Nobel Prizes in both physics and chemistry', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'science'},
        {'question': 'How do the Amazon River and the Nile River compare in terms of length and geographical significance?', 'answer': 'The Amazon is the longest river flowing through South America, while the Nile was crucial to ancient Egyptian civilization', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'geography'},
        {'question': 'What connects the Industrial Revolution with modern technological advancement?', 'answer': 'The Industrial Revolution began mechanization that evolved into today\'s technological automation and digitization', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'history-technology'},
        {'question': 'How do neural networks relate to both biological inspiration and artificial intelligence?', 'answer': 'Neural networks are AI models inspired by biological neural networks in the brain', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'technology-science'},
        {'question': 'What connects Galileo\'s observations with our understanding of the solar system?', 'answer': 'Galileo used telescopic observations to support the heliocentric model, changing our view of Earth\'s place in the universe', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'science-history'},
        {'question': 'How do blockchain technology and traditional database systems differ in their approach to data integrity?', 'answer': 'Blockchain uses distributed cryptographic verification while traditional databases rely on centralized control for data integrity', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'technology'},
        {'question': 'What connects Shakespeare\'s literary contributions with the broader Renaissance cultural movement?', 'answer': 'Shakespeare exemplified Renaissance ideals through innovative dramatic works that explored human nature and classical themes', 'type': 'multi-hop', 'difficulty': 'hard', 'domain': 'arts-history'},
        
        # Very Hard Questions - Complex Conceptual Reasoning (20 questions)
        {'question': 'How do artificial intelligence, machine learning, and deep learning relate to each other in a hierarchical structure?', 'answer': 'AI is the broadest field, machine learning is a subset of AI, and deep learning is a specialized subset of machine learning using neural networks', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'technology'},
        {'question': 'What cultural and historical significance connects the Renaissance period with artists Leonardo da Vinci and Michelangelo?', 'answer': 'The Renaissance was a period of cultural rebirth that both Leonardo and Michelangelo exemplified through revolutionary artistic innovations and interdisciplinary work', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'arts-history'},
        {'question': 'How do the scientific contributions of Einstein, Newton, and Darwin each represent paradigm shifts in different fields?', 'answer': 'Einstein revolutionized physics with relativity theory, Newton established classical mechanics, and Darwin transformed biology with evolution theory', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'science'},
        {'question': 'What role do programming languages like Python and JavaScript play in the broader context of artificial intelligence and web technologies?', 'answer': 'Python serves as a primary language for AI development, while JavaScript enables AI integration in web applications and user interfaces', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'technology'},
        {'question': 'How do the Olympic Games represent the intersection of ancient culture, modern international cooperation, and athletic excellence?', 'answer': 'The Olympics bridge ancient Greek traditions with modern global unity, showcasing human achievement while promoting international peace', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'sports-history'},
        {'question': 'What parallels exist between biological processes like photosynthesis and technological systems like artificial intelligence?', 'answer': 'Both involve systematic conversion of inputs to outputs: photosynthesis converts light to chemical energy, AI converts data to intelligent outputs', 'type': 'cross-domain', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'How do geographical landmarks like the Great Wall of China and modern technological networks like the Internet represent human connectivity?', 'answer': 'Both represent massive human efforts to connect across distances - the Great Wall for physical defense, the Internet for global communication', 'type': 'cross-domain', 'difficulty': 'very_hard', 'domain': 'geography-technology'},
        {'question': 'What connections exist between Renaissance cultural innovation and modern technological innovation in terms of human creativity?', 'answer': 'Both periods represent explosions of human creativity - Renaissance combined art and science, modern era combines technology with human-centered design', 'type': 'cross-domain', 'difficulty': 'very_hard', 'domain': 'arts-technology'},
        {'question': 'How do Marie Curie\'s interdisciplinary achievements parallel modern approaches in computer vision that combine multiple fields?', 'answer': 'Both demonstrate interdisciplinary success - Curie bridged physics and chemistry, computer vision combines AI, mathematics, and engineering', 'type': 'analogical', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'What systematic approaches do both Roman governance and blockchain technology represent in terms of record-keeping?', 'answer': 'Both created systematic methods for trusted records - Romans through centralized administration, blockchain through decentralized verification', 'type': 'analogical', 'difficulty': 'very_hard', 'domain': 'history-technology'},
        {'question': 'How do the evolutionary principles discovered by Darwin relate to modern machine learning algorithms that adapt and improve?', 'answer': 'Both involve iterative improvement through selection - evolution through natural selection, machine learning through algorithmic optimization', 'type': 'conceptual', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'What connects the revolutionary impact of the printing press with modern Internet technologies in information dissemination?', 'answer': 'Both democratized information access - printing press made books widely available, Internet makes global information instantly accessible', 'type': 'historical-parallel', 'difficulty': 'very_hard', 'domain': 'history-technology'},
        {'question': 'How do quantum mechanical principles challenge classical physics similarly to how AI challenges traditional computing?', 'answer': 'Both represent paradigm shifts - quantum mechanics introduced probabilistic vs deterministic physics, AI introduces adaptive vs programmed computing', 'type': 'paradigm-shift', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'What role do rivers like the Amazon and Nile play in their respective ecosystems compared to data flows in computer networks?', 'answer': 'Both serve as critical transportation systems - rivers carry nutrients and enable life, networks carry information and enable digital communication', 'type': 'system-analogy', 'difficulty': 'very_hard', 'domain': 'geography-technology'},
        {'question': 'How do the collaborative aspects of the Human Genome Project parallel modern open-source software development?', 'answer': 'Both involve global collaboration, shared knowledge, and collective problem-solving to achieve breakthrough innovations', 'type': 'collaboration-model', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'What connections exist between the artistic techniques of Renaissance masters and modern computer graphics algorithms?', 'answer': 'Both involve mathematical principles for realistic representation - Renaissance perspective geometry, computer graphics 3D modeling and rendering', 'type': 'technique-evolution', 'difficulty': 'very_hard', 'domain': 'arts-technology'},
        {'question': 'How do the navigation methods used by ancient explorers relate to modern GPS and satellite technology?', 'answer': 'Both solve the fundamental problem of location determination - ancient methods used stars and landmarks, modern GPS uses satellite triangulation', 'type': 'problem-solving-evolution', 'difficulty': 'very_hard', 'domain': 'history-technology'},
        {'question': 'What parallels exist between the immune system\'s pattern recognition and machine learning\'s classification algorithms?', 'answer': 'Both identify patterns to distinguish between normal and abnormal - immune systems detect pathogens, ML algorithms classify data patterns', 'type': 'biological-computational', 'difficulty': 'very_hard', 'domain': 'science-technology'},
        {'question': 'How do the preservation methods used for ancient handwritten records compare to modern data backup and archival systems?', 'answer': 'Both address information preservation across time - handwritten records through physical protection, digital systems through redundant storage and format migration', 'type': 'preservation-methods', 'difficulty': 'very_hard', 'domain': 'history-technology'},
        {'question': 'What connections exist between the democratic ideals of ancient Athens and modern collaborative platforms like Wikipedia?', 'answer': 'Both embody collective knowledge creation and democratic participation in information sharing and decision-making processes', 'type': 'governance-information', 'difficulty': 'very_hard', 'domain': 'history-technology'}
    ]
    
    logger.info(f"Created large-scale dataset: {len(corpus)} documents, {len(questions)} questions")
    
    # Comprehensive analysis
    difficulty_counts = defaultdict(int)
    type_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    
    for q in questions:
        difficulty_counts[q['difficulty']] += 1
        type_counts[q['type']] += 1
        domain_counts[q['domain']] += 1
    
    logger.info(f"Difficulty distribution: {dict(difficulty_counts)}")
    logger.info(f"Question types: {dict(type_counts)}")
    logger.info(f"Domain coverage: {dict(domain_counts)}")
    
    return corpus, questions
