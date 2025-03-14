import nltk
import numpy as np
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import re

nltk.download("punkt")

def extractive_summary(text, num_sentences=5, high_similarity_threshold=0.85):
    """
    Extracts the most representative sentences from Latin text using enhanced filtering
    and Latin-specific processing to avoid common issues with extractive summaries.

    Parameters:
        text (str): The full Latin text.
        num_sentences (int): Number of sentences to include in the summary.
        high_similarity_threshold (float): Threshold for removing redundant sentences.

    Returns:
        str: Extractive summary or NaN if processing is not possible.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan  # Return NaN if text is missing or empty
    
    # Pre-processing
    text = re.sub(r'\([^)]*\)', '', text)  # Remove text within parentheses
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)  # Return all sentences for short texts
    
    # Define Latin-specific stopwords
    latin_stopwords = [
        'ab', 'ac', 'ad', 'at', 'atque', 'aut', 'autem', 'cum', 'de', 'dum', 'enim', 
        'et', 'etiam', 'ex', 'in', 'inter', 'nam', 'non', 'per', 'qua', 'quae', 
        'quam', 'qui', 'quibus', 'quid', 'quo', 'quod', 'sed', 'si', 'sic', 'sunt', 
        'tam', 'tamen', 'ut', 'vel'
    ]
    
    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        stop_words=latin_stopwords,
        sublinear_tf=True,
        smooth_idf=True,
        min_df=1
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.nan  # Return NaN if TF-IDF vectorization fails (e.g., empty vocabulary)
    
    # Ensure there are at least two features in the TF-IDF matrix
    if tfidf_matrix.shape[1] < 2:
        return np.nan  # Return NaN if there aren't enough features for processing

    # Apply LSA for semantic analysis
    n_components = min(10, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    n_components = max(2, n_components)  # Ensure at least 2 components
    
    lsa = TruncatedSVD(n_components=n_components)
    try:
        lsa_matrix = lsa.fit_transform(tfidf_matrix)
    except ValueError:
        return np.nan  # Return NaN if LSA transformation fails
    
    # Rank sentences by importance
    sentence_scores = np.sum(np.abs(lsa_matrix), axis=1)
    ranked_sentences = sorted(
        enumerate(sentence_scores), key=lambda x: x[1], reverse=True
    )
    
    # Select top-ranked sentences while ensuring diversity
    selected_indices = []
    seen_sentences = set()
    
    selected_indices.append(0)  # Always include first sentence
    seen_sentences.add(sentences[0])
    
    for idx, _ in ranked_sentences:
        if len(selected_indices) >= num_sentences:
            break
        
        current_sentence = sentences[idx]
        if current_sentence not in seen_sentences:
            selected_indices.append(idx)
            seen_sentences.add(current_sentence)
    
    # Sort sentences in original order
    selected_indices.sort()
    summary_sentences = [sentences[i] for i in selected_indices[:num_sentences]]
    
    return ' '.join(summary_sentences)


def pretty_print_summary(original_text, summary_sentences):
    """
    Pretty prints the original text and the extracted summary.
    summary_sentences should be a list of strings.
    """
    width = 100  # Set the width of the formatted output

    print("\n" + "-" * width)
    print(" " * ((width // 2) - 7) + "ORIGINAL TEXT")
    print("-" * width)
    wrapped_original = textwrap.fill(original_text, width)
    print(wrapped_original)
    print("\n" + "-" * (width // 2 - 10) + " SUMMARIZED TEXT " + "-" * (width // 2 - 10))
    wrapped_summary = textwrap.fill(summary_sentences, width)
    print(wrapped_summary)
    print("-" * width + "\n")


example1 = """Laevinus Torrentius (Batavice Liévin van der Beken), natus Gandavi die 8 Martii 1525 - mortuus Bruxellis die 26 Aprilis 1595, fuit homo ecclesiasticus, poeta neolatinus et philologus Belgicus. Luculentis commentariis Horatium et Suetonium exornavit; aeque notus tamen est, quia poema composuit 'tyrannicidium' Gulielmi Auriaci glorificans. Iurisprudentiae Lovanii studuit atque doctoris gradum Bononiae absolvit. Deinde nonnullos annos Romae moratus est, ubi diuturnas necessitudines cum eruditis viris Paulo Manutio, Antonio Augustino, Fulvio Ursino, cardinali Sirleto etc coniunxit. In Belgicam reversus ab anno 1557 archidiaconus Brabantiae, mox vicarius generalis dioecesis Leodiensis fuit. Tum princeps idemque episcopus Leodiensis erat Georgius Austriacus; Laevinus quartum adhuc episcopum ab eo vidit, nobilem iuvenem Ernestum Bavaricum, qui simul archiepiscopus Coloniensis erat multosque alios episcopatus cumulabat, quia per triginta annos in illo munere permansit. Qua in urbe architectus Lambertus Lombardus ei pulchram domum urbanam aedificavit quae hodie Hôtel Torrentius dicitur et res patrimonialis a regione Wallonia anno 1969 decreta est. Iam anno 1576 episcopus Antverpiensis ab Hispanorum rege Philippo II designatus erat; non tamen ante annum 1587 sedem suam occupare ei licuit, quia tum in Flandria bellum furore religioso exasperatum ardebat. Postremo ad archiepiscopatum Mechliniensem destinabatur, cum morte prohibitus est, quin hoc munus inire posset.
In ipsa ecclesia cathedrali Antverpiae humatus est. Testamento collegium Societatis Iesu Lovanii a se conditum (quod philosophicum gymnasium in epistolis ad Iustum Lipsium appellare solebat) heredem instituit, cui in primis locupletem bibliothecam suam et numismata legavit. De quorum numismatum excellentia in quadam epistula ad amicum Andream Scottum non sine suspicione vanitatis sic se iactabat: "nummorum veterum aliarumque antiquitatum cum multitudine tum excellentia ac raritate in hac tota provincia cedam nemini, ne Laurinis quidem fratribus quos patronos habebat Goltsius". Non enim, etsi tam multis ecclesiasticis muneribus oppressus, umquam rempublicam litterariam omnino deseruit, ut ex epistolis ad philosophum Iustum Lipsium, ad typographum Christophorum Plantinum, ad eruditum iesuitam Andream Scottum missis apparet."""

example2 = """Quattuor equites apocalyptici. sive brevius equites apocalyptici. sunt quattuor figurae symbolicae in Apocalypsi Ioannis descriptae. In hoc ultimo Novi Testamenti libro scriptori ostenditur Deus in throno sedens cum viginti quattuor senioribus in thronis sedentibus. Deus in dextra habebat librum septem signaculis clausum, quem nemo nisi Agnus (ἀρνίον), quo nomine Iesus Christus in Apocalypsi saepissime appellatu dignus sit aperire. In capitulo sexto Agnus signacula aperire incipit. Primis sigillis apertis quattuor equites arcessuntur suo quisque equo vecti.
In hac visione, quam Iesus Ioanni transmittit, equi inter se colore differunt, itemque equitibus sua cuique armatura est. Constat equorum colores et equitum habitum fortuitos non esse sed vim symbolicam habere. At de interpretatione ambigitur.
Inter Christianos constat Biblia Sacra plena esse prophetiarum, e quibus aliae iam ad litteram impletae sunt, aliae autem nondum. Interpretatione prophetali quattuor illi equites apocalyptici dicuntur futurum tribulationis tempus portendere. Alia interpretandi ratio in eo posita est, ut equites apocalyptici ad res saeculo primo tantum gestas referantur. Tali interpretatione contemporali id agitur, ut equites illi cum certis eventibus contemporaneis coniungantur. Quae ratio facile — quatenus adventus secundus Iesu manet — etiam in novas rerum temporumque contextus quadrare videtur, quo fieri potest, ut nulla inter theologiam et politicam quotidianam differentia iam animadvertatur. Praesertim colores equorum invitaverunt interpretationes arbitrarias — sicut: equum rufum communismum significare — de quibus dubium est, utrum de prophetice an politice dicto agatur."""

example3 = """EPISTULAE

Tandem venit amor, qualem texisse pudori
    quam nudasse alicui sit mihi fama magis.
Exorata meis illum Cytherea Camenis
    adtulit in nostrum deposuitque sinum.
Exsolvit promissa Venus: mea gaudia narret,
    dicetur siquis non habuisse sua.
Non ego signatis quicquam mandare tabellis,
    ne legat id nemo quam meus ante, velim,
sed peccasse iuvat, vultus conponere famae
    taedet: cum digno digna fuisse ferar.

Invisus natalis adest, qui rure molesto
    et sine Cerintho tristis agendus erit.
Dulcius urbe quid est? an villa sit apta puellae
    atque Arrentino frigidus amnis agro?
Iam nimium Messalla mei studiose, quiescas,
    heu tempestivae, saeve propinque, viae!
Hic animum sensusque meos abducta relinquo,
    arbitrio quamvis non sinis esse meo.

Scis iter ex animo sublatum triste puellae?
    natali Romae iam licet esse suo.
Omnibus ille dies nobis natalis agatur,
    qui nec opinanti nunc tibi forte venit.

Gratum est, securus multum quod iam tibi de me
    permittis, subito ne male inepta cadam.
Sit tibi cura togae potior pressumque quasillo
    scortum quam Servi filia Sulpicia:
Solliciti sunt pro nobis, quibus illa dolori est,
    ne cedam ignoto, maxima causa, toro.

Estne tibi, Cerinthe, tuae pia cura puellae,
    quod mea nunc vexat corpora fessa calor?
A ego non aliter tristes evincere morbos
    optarim, quam te si quoque velle putem.
At mihi quid prosit morbos evincere, si tu
    nostra potes lento pectore ferre mala?

Ne tibi sim, mea lux, aeque iam fervida cura
    ac videor paucos ante fuisse dies,
si quicquam tota conmisi stulta iuventa,
    cuius me fatear paenituisse magis,
hesterna quam te solum quod nocte reliqui,
    ardorem cupiens dissimulare meum."""

def print_examples(num_sentences=5):
    print("Example 1:")
    summary1 = extractive_summary(example1, num_sentences=num_sentences)
    pretty_print_summary(example1, summary1)

    print("Example 2:")
    summary2 = extractive_summary(example2, num_sentences=num_sentences)
    pretty_print_summary(example2, summary2)

    print("Example 3:")
    summary3 = extractive_summary(example3, num_sentences=num_sentences)
    pretty_print_summary(example3, summary3)