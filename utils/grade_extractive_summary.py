import torch
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download
import re
import nltk
import os
# Additional libraries for metrics
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


###############################################################################
# EXAMPLE DATASETS & EXAMPLES FOR FEW-SHOT
###############################################################################

examples = {
    'texts' : ['Bibliotheca publica est bibliotheca quae, omnibus lectoribus patens, pecuniis publicis sicut vectigalibus plerumque exercetur. Institutio est quam operantur bibliothecarii operariique professionales qui etiam servi civiles sunt. Bibliothecis publicis sunt quinque proprietates fundamentales: vectigalibus sustinentur (plerumque ipsius loci, quamquam ullus rectionis gradus contribuere potest); a consilio administrantur ut rebus publicis faveant; omnibus patent, et quisque socius communitatis collectioni admitti potest; omnino voluntariae sunt, et nemo ad utilitatem ex eis capiendam compelli potest; atque officia fundamentalia sine impensá praebent. Bibliothecae publicae in multis civitatibus inveniuntur, ubi necessaria multitudinis educatae litterataeque pars saepe habentur. Bibliothecae publicae a bibliothecis investigationis, bibliothecis scholasticis, aliisque bibliothecis praecipuis distinctae sunt, quia earum propositum est data omnibus, potius quam certis scholis, institutionibus, multitudiníve investigatoriae, praebere. Etiam officia libere praebent, sicut praescholasticas fabularum horas ad alphabetismum inter puerulis cohortandum, locos tacitos studii laborisque discipulis et hominibus professionalibus, ac sodalitates librorum ad amorem litterarum in adultis alendum. Bibliothecae publicae usitate sinunt ut utentes libros aliasque materias mutuentur, eas ex praedio transferentes; praeterea collectiones referentiae quae in usum non veniunt habent, atque accessum ad computatra et interrete utentibus praebent. Historia prima. Bibliotheca Malatestiana, prima bibliotheca a civitate operata, Caesenae in Italia anno 1447 constituta, textus ambo religiosas profanasque Latine, Graece, Hebraice scriptas suppeditabat, omnibusque omnino patebat. Claudius Sallier, clericus et philologus Francicus, bibliothecam publicam Sedoloci, in oppido comitatus Burgundae et Liberi, ab anno 1737 ad annum 1750 operabatur. Iosephus Andreas Zaluscius et Andreas Stanislaus Zaluscius, fratres episcopique Catholici, Bibliothecam Zaluscianam Varsoviae ab 1747 ad 1795 operabantur, quae bibliotheca omnibus patebat et prima bibliotheca publica Polonica fuit, maxima quoque in Polonia, atque una ex primis bibliothecis publicis in Europa constitutis. Nexus externi.',
 'Erhardus Hübener Res apud repertae: Erhardus Fridericus Iulius ("Erhard Friedrich Julius") Hübener (natus 4 Augusti 1881 in vico Tacken (Brandenburgum), mortuus 3 Iunii 1958 in oppido Bad Salzuflen (Rhenania Septentrionalis-Vestfalia)) vir publicus Germaniae et sodalis primo DDP, deinde LDPD fuit. Iuventus et munus. Pater eius pastor protestans fuit. Anno 1901 maturitatem adeptus est et deinde Kieliae et Berolini oeconomiae studebat atque anno 1905 doctor promotus est. Deinde mercator laborabat et primo bello mundano dux militum interfuit. Ab anno 1919 in administerio Borussiae commercii minister publicus laboravit. Cursus honorum. Hübener, qui aber anno 1919 sodalis DDP erat, anno 1922 praepositus vicarius et anno 1924 praepositus ("Landeshauptmann") provinciae Borussicae Saxoniae electus est. Hoc munere fungens terram novam foederalem Saxoniam-Anhaltinum proposuit, sed frustra. Anno 1930 iterum electus est, sed anno 1933 a nazistis magistratu dimissus est. Cum inter dictaturam nazistarum e vita publica recessisset, post secundum bellum mundanum ab auctoritatibus Unionis Sovieticae, quae hanc partem Germaniae occupaverant, iterum praepositus provinciae Saxoniae factus est. 3 Decembris 1946 praeses ministrorum Saxoniae-Anhaltini novae terrae e provincia creatae electus est. Cum magistratu fungeretur, semper contra Germaniam inter victores divisam certabat, sed frustra: Neque praesides ministrorum occidentalium Zonarum neque Unio Sovietica eum hoc consilio adiuvabant. Praeterea hoc tempore nova dictatura SED in oriente parte Germaniae orta est: Multi ministri factionum liberalium et CDU a communistis dimissi sunt. Itaque Hübener 1 Octobris 1949 e vita publica recessit. Ultimi anni. Hübener nunc professor administrationis universitatis Salinarum Saxonicarum factus est. Anno 1958 ad curationem iter in partem occidentalem Germaniae fecit et ibi 3 Iunii 1958 mortuus est. Nexus externi. Praesides Ministrorum Saxoniae-Anhaltini',
 '"Opisthocomus hoazin" in Viridarro Nationali Peruviae ------ Animalia — Chordata Classis : Aves Infraclassis : Neognathae Ordo : Opisthocomiformes Familia : Opisthocomidae Genus : Opisthocomus Species : Opisthocomus hoazin Palaeontologia Miocaeno–Recens Conservationis status LCVigens Synonyma "Phasianus hoazin" Territorium Distributio geographica Opisthocomus hoazin est species avium tropicarum familiae Opisthocomidarum. In paludinibus, silvis riparianis, manglibusque Amazoniae et Orinoci in America Australi invenitur. Pullis insigniter sunt ungulae in duobus digitis alarum. "Opisthocomus hoazin" est sola species generis Opisthocomi (Graece \'crines postici longi\', ad magnam cristam attingens), quod vicissim est solum suae familiae genus. Taxinomicus autem huius familiae status, ab eruditis diu et vehementer disputatus, nondum est certus. Musophagidae, aves ordinis Musophagiformium convergenter evolutae, etiam magnas cristas habentes, sunt arboricolae et herbivorae, quarum pulli etiam ungulis scandentes utuntur. "Opisthocomus hoazin" est avis civica Guianae. Descriptio. "Opisthocomus hoazin" est avis magnitudine phasianinarum, 65 centimetra longi, collo longo et capite parvo. Facies est caerulea sine pinnis, oculis marroninis, et supra caput est crista spicata et rufa. Cauda, longa, fusca, fuliginea, in apice bubalino late finitur. Partes superiores sunt bubalinae obscurae in tegminibus alarum, et bubalinae fasciatae in armis et cervice. Partes inferiores sunt bubalinae, sed crissum (sub cauda) cloacam circumcludit). Pennae primariae, tegmina sub alis, et ilia sunt rufa et castanea clara, sed hoc videri potest plerumque cum alas aperiat. "Opisthocomus hoazin"" est herbivorus, qui foliis fruticibusque vescitur. Nexus externi.',
 'Petrus Skarga Res apud repertae: Petrus Skarga (Scarga), fuit scriptor, theologus-polemista, praedicator regius, actuosus in re sociali et in reformatione catholica, primus rector Academiae Vilnensis. Natus die 2 Februarii 1536 in Grójec ad Varsaviam, mortuus die 27 Septembris 1612 Cracoviae. Vita. Annis 1552-1555 studuit in Academia Cracoviensi. Postea rector fuit scholae parochialis ad ecclesiam S. Joannis Varsaviae (1555–1557). A. 1564 Leopoli presbyter ordinatus est. A. 1568 Romam petit, ubi a. 1569 Societatem Jesu ingressus est. Deinde in Collegio Romano theologiae studuit. In Poloniam reversus, primum Pultoviae (Pułtusk) moratus est (1571-1572), postea Vilnam missus est (1573), ubi primus rector Academiae Vilnensis fuit (1579–1584). Postea Cracoviae praepositus fuit Domus Professae S. I. ad S. Barbarae, ubi a. 1584 Confraternitatem Misericordiae (existit usque hodie) necnon tabernam bancariam aliaque opera ad pauperes iuvandos fundavit. Confraternitatem Misericordiae etiam in aliis Regni civitatibus promovit. Ab a. 1588 ad a. 1612 contionator fuit regis Sigismundi III. Anno 1612 venit Cracoviam. Mortuus est die 27 Septembris, fama sanctitatis fruens. Sepultus est in ecclesia SS. Apostolorum Petri et Pauli. Opera. Skarga multa scripsit de variis materiis. Libri principales sunt sequentes: Omnes fere libri plures habuerunt editiones posteriores, etiam nunc temporis. Nonnulla eius scripta, praesertim "Vitae sanctorum" et "Contiones in parlamento" in alias linguas versa sunt, ex. gratia "Les sermons politiques", Parisiis 1916. Merita. Skarga notabilem influxum in vitam politicam et socialem Poloniae et Lithuaniae tunc temporis exercuit. Ad unionem Brestensem a. 1596 inter catholicos et partem orthodoxorum in Polonia viventium multum contulit. De excolenda lingua Polonica optime meritus est. Processus beatificationis Petri Skarga, in Archidioecesi Cracoviensi annis 2013-2016 peractus, die 21 Junii 2016 a. conclusus est. Reliqua in Congregatione Vaticana de Causis Sanctorum fient.',
 'Publius Iuventius Celsus Titus Aufidius Hoenius Severianus (nescimus, quando natus mortuusve sit) senator Romanus et iuris consultus primo et secundo saeculo fuit. Vita et cursus honorum. Filius et discipulus Publii Iuventii Celsi Patris iuris consulti fuit et in Italia septentrionali natus est. Accepit cum Lucio Neratio Prisco scholam iuris Iuventii patris. Consiliarius Hadriani imperatoris factus anno 129 consul ordinarius cum filio Neratii Prisci et postea proconsul Asiae nominatus est. Celsus consul "senatus consulto Iuventiano" favebat, quo heres credulus hereditatis caducae non plus quod lucri fecerat, deberet. Quod de iure sensuit. Celsus numquam ius scriptum interpretabatur, sed semper quaesivit, quid sit natura iuris. Itaque saepe ius traditum iustitiae causa mutavit. Exemplum notum condictio Iuventiana est (Cels. (6 D.) Dig. XII 1,32). Problema iurisprudentiae redditionem aeris alienae agere est. Tres vires, Tu et Titius et Ego, Tu obligatum rogavit, quae pecuniam creditam reddere. Dig. XII.1.32: "Si et me et Titium mutuam pecuniam rogaveris et ego meum debitorem tibi promittere iusserim, tu stipulatus sis, cum putares eum Titii debitorem esse, an mihi obligaris? subsisto, si quidem nullum negotium mecum contraxisti: sed propius est, ut obligari te existimen, non quia pecuniam tibi credidi (hoc enim nisi inter consentientes fieri non potest): sed quia pecunia mea quae ad te pervenit, eam mihi a te reddi bonum et aequum est."" Controversum est causa obligationis. Presumitur actionem condicendi iustam esse (Dig XII 1 de condictionibus agit) quamquam negotium contractum abest (Julian (Dig. 12.6.33) "c"ondictio .. non habebit, quia nullum negotium inter nos contraheretur"). Doctores historiae iuris diputandum, obligatio ex re vel iure naturae oritur. Casus belli sententia ultima responsae: ""sed quia pecunia mea quae ad te pervenit, eam mihi a te reddi bonum et aequum est"." Alii fundamentum conclusionis pars sententae prima (..."pervenit"), alii ratio decidendi aequitas iuris gentium putant. Dicta. Nonnulla dicta clara Celsi nobis tradita sunt:'],
    'summaries': [['Bibliothecis publicis sunt quinque proprietates fundamentales: vectigalibus sustinentur ; a consilio administrantur ut rebus publicis faveant; omnibus patent, et quisque socius communitatis collectioni admitti potest; omnino voluntariae sunt, et nemo ad utilitatem ex eis capiendam compelli potest; atque officia fundamentalia sine impensá praebent.',
  'Bibliothecae publicae in multis civitatibus inveniuntur, ubi necessaria multitudinis educatae litterataeque pars saepe habentur.',
  'Bibliothecae publicae a bibliothecis investigationis, bibliothecis scholasticis, aliisque bibliothecis praecipuis distinctae sunt, quia earum propositum est data omnibus, potius quam certis scholis, institutionibus, multitudiníve investigatoriae, praebere.',
  'Etiam officia libere praebent, sicut praescholasticas fabularum horas ad alphabetismum inter puerulis cohortandum, locos tacitos studii laborisque discipulis et hominibus professionalibus, ac sodalitates librorum ad amorem litterarum in adultis alendum.',
  'Bibliotheca Malatestiana, prima bibliotheca a civitate operata, Caesenae in Italia anno 1447 constituta, textus ambo religiosas profanasque Latine, Graece, Hebraice scriptas suppeditabat, omnibusque omnino patebat.'],
 ['Pater eius pastor protestans fuit.',
  'Deinde mercator laborabat et primo bello mundano dux militum interfuit.',
  'Ab anno 1919 in administerio Borussiae commercii minister publicus laboravit.',
  'Praeterea hoc tempore nova dictatura SED in oriente parte Germaniae orta est: Multi ministri factionum liberalium et CDU a communistis dimissi sunt.',
  'Anno 1958 ad curationem iter in partem occidentalem Germaniae fecit et ibi 3 Iunii 1958 mortuus est.'],
 ['"Opisthocomus hoazin" in Viridarro Nationali Peruviae ------ Animalia — Chordata Classis : Aves Infraclassis : Neognathae Ordo : Opisthocomiformes Familia : Opisthocomidae Genus : Opisthocomus Species : Opisthocomus hoazin Palaeontologia Miocaeno–Recens Conservationis status LCVigens Synonyma "Phasianus hoazin" Territorium Distributio geographica Opisthocomus hoazin est species avium tropicarum familiae Opisthocomidarum.',
  'Taxinomicus autem huius familiae status, ab eruditis diu et vehementer disputatus, nondum est certus.',
  '"Opisthocomus hoazin" est avis magnitudine phasianinarum, 65 centimetra longi, collo longo et capite parvo.',
  'Facies est caerulea sine pinnis, oculis marroninis, et supra caput est crista spicata et rufa.',
  'Pennae primariae, tegmina sub alis, et ilia sunt rufa et castanea clara, sed hoc videri potest plerumque cum alas aperiat.'],
 ['Anno 1612 venit Cracoviam.',
  'Apostolorum Petri et Pauli.',
  'Nonnulla eius scripta, praesertim "Vitae sanctorum" et "Contiones in parlamento" in alias linguas versa sunt, ex.',
  'Skarga notabilem influxum in vitam politicam et socialem Poloniae et Lithuaniae tunc temporis exercuit.',
  'Processus beatificationis Petri Skarga, in Archidioecesi Cracoviensi annis 2013-2016 peractus, die 21 Junii 2016 a. conclusus est.'],
 ['Filius et discipulus Publii Iuventii Celsi Patris iuris consulti fuit et in Italia septentrionali natus est.',
  'Itaque saepe ius traditum iustitiae causa mutavit.',
  'Controversum est causa obligationis.',
  'Presumitur actionem condicendi iustam esse  quamquam negotium contractum abest  "c"ondictio .. non habebit, quia nullum negotium inter nos contraheretur").',
  'Nonnulla dicta clara Celsi nobis tradita sunt:']],
    'grades': ['53', '29', '71', '84', '9'],
'explanations': [
    "The summary captures the main idea of public libraries but lacks depth. It fails to convey the historical progression and societal impact, making it a poor representation of the original. Additionally, the **logical flow is weak**—sentences jump between broad descriptions and specific details without smooth transitions. This disrupts readability and makes it harder to understand how public libraries evolved.",
    
    "The summary is **highly fragmented**. While it mentions that Hübener was a German politician, it does not properly connect the key moments of his career. The **chronological order is unclear**, making it difficult to follow his journey from his early career, through Nazi oppression, to his later academic life. Important transitions are missing, and some facts appear in isolation, reducing coherence.",
    
    "The summary captures the species identification but **fails at presenting a structured analysis**. Details about taxonomy, evolution, and behavior appear **disjointed**—there is no clear logical flow between ideas. For instance, it jumps from a physical description to taxonomy, and then to its ecological niche without smooth transitions, making it difficult to grasp the key takeaways.",
    
    "This summary is structurally decent, with a clear introduction to Skarga’s role. However, it **lacks logical connections between his achievements**. The discussion on his theological impact feels disconnected from his academic contributions, making the summary feel like a collection of facts rather than a well-structured argument. Stronger **cause-effect linking** would improve readability.",
    
    "The summary is **extremely weak and incoherent**. While it acknowledges Celsus' identity, it **jumps erratically between his background, legal contributions, and philosophy without a clear structure**. There is no guiding logic that connects his influence on Roman law with his interpretations of justice. This makes it nearly impossible to extract a meaningful understanding from the text."
]

}

###############################################################################
# MODEL LOADING
###############################################################################

def load_model(
    MODEL_PATH="/Data/AxelDlv/Mistral-7B-Instruct-v0.3",
    tokenizer_name="tokenizer.model.v3",
    DOWNLOAD_MODEL=False
):
    """
    Load Mistral model and tokenizer from the specified path.
    If DOWNLOAD_MODEL=True, it will pull them from huggingface_hub first.
    """
    if DOWNLOAD_MODEL:
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            allow_patterns=[
                "params.json",
                "consolidated.safetensors",
                "tokenizer.model.v3"
            ],
            local_dir=MODEL_PATH,
        )
    
    tokenizer_path = os.path.join(MODEL_PATH, tokenizer_name)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = MistralTokenizer.from_file(tokenizer_path)
    model = Transformer.from_folder(MODEL_PATH)
    return model, tokenizer

def free_model(model, tokenizer):
    """
    Cleanly free GPU memory if needed.
    """
    del model
    del tokenizer
    torch.cuda.empty_cache()

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def concatenate_sentences(sentences):
    """
    Utility to join an array of sentences with a space.
    """
    return ' '.join(sentences)

###############################################################################
# LLM-BASED SCORE EVALUATION
###############################################################################

def evaluate_summary(original_text, summary_text, model, tokenizer, temperature=0.2, n_tokens=1):
    """
    Evaluate an extractive summary via Mistral LLM using a few-shot prompt.
    Returns a numeric grade (0-100) along with an explanation.
    """
    
    # Build few-shot examples with explanations
    few_shot_section = ""
    for i in range(len(examples['texts'])):
        few_shot_section += (
            f"**Example {i+1}:**\n"
            f"**Original Text:**\n{examples['texts'][i]}\n"
            f"**Summary:**\n{concatenate_sentences(examples['summaries'][i])}\n"
            f"**Grade:** {examples['grades'][i]}/100\n"
            f"**Explanation:** {examples['explanations'][i]}\n\n"
        )

    # Construct the final prompt
    prompt = (
        f"""[INST] You are an **expert evaluator of extractive summaries** with **no tolerance for mediocrity**. Your **sole duty** is to **assign a score out of 100** that reflects the summary’s **true quality**. Do not hesitate to **assign very low scores** if necessary.
Your evaluation **must be merciless** and based on two fundamental pillars:

### **Logical Coherence & Structural Flow (Max: 50 points)**
✔ **Does the summary present information in a clear, logical order?**  
✔ **Are ideas connected smoothly, or does it jump between unrelated points?**  
✔ **Does it properly introduce, elaborate, and conclude key ideas?**  
✔ **Does it avoid abrupt or missing transitions between sections?**  
✔ **Is the summary understandable without the original text ?** 
✔ **Do some sentences start from nowhere or end abruptly or refer to something that is not explained?**

### **Content Accuracy & Depth (Max: 50 points)**
✔ **Does the summary preserve essential details from the original text?**  
✔ **Is it concise yet sufficiently detailed to cover the topic?**  
✔ **Does it capture the key themes and arguments?**

### **Scoring Guidelines**  
**90-100:** A rare, flawless summary. **Highly structured**, clear, and deeply informative.  
**70-89:** Strong but flawed. **Some lapses in flow or minor structural issues**.  
**50-69:** Mediocre. **Noticeable gaps, unclear logical transitions, or missing context**.  
**20-49:** Poor. **Ideas appear in random order, key connections are missing, and structure is chaotic**.  
**0-19:** **Unacceptable**—the summary is **nonsensical, highly misleading, or completely disorganized**.  

### **Examples of Evaluations**
{few_shot_section}

### **Now evaluate the following summary:**

**Original Text:**\n{original_text}
**Extractive Summary:**\n{summary_text}
**Score:** [/INST]""")

    # Encode the prompt into tokens
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    # Generate a response. Expecting both the score and explanation.
    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=n_tokens,  # Allow more tokens for a detailed critique
        temperature=temperature,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )

    # Decode the response from tokens
    response = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]).strip()

    return response

def convert_output_to_grade(output):
    """
    Convert the Mistral LLM output to a grade (0-100).
    """
    # Extract the score from the output
    match = re.search(r"\d{1,3}", output) # Match 1-3 digits
    grade = float(match.group(0)) if match else 0
    return grade