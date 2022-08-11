import random
from dateutil import parser
random_text_field = [
    "arnie morton \s of chicago", "art \s delicatessen", "hotel bel-air", "cafe bizou", "campanile", "chinois on main",
    "citrus", "fenix", "granita", "grill on the alley", "restaurant katsu", "l \ orangerie", "le chardonnay",
    "locanda veneta", "matsuhisa", "the palm", "patina", "philippe \s the original", "pinot bistro",
    "rex il ristorante", "spago", "valentino", "yujean kang \s gourmet chinese cuisine", "'21 club", "aquavit",
    "aureole", "cafe lalo", "cafe des artistes", "carmine \s", "carnegie deli", "chanterelle", "daniel", "dawat",
    "felidia", "four seasons grill room", "gotham bar & grill", "gramercy tavern", "island spice", "jo jo",
    "la caravelle", "la cote basque", "le bernardin", "les celebrites", "lespinasse", "lutece", "manhattan ocean club",
    "march", "mesa grill", "mi cocina", "montrachet", "oceana", "park avenue cafe", "petrossian", "picholine", "pisces",
    "rainbow room", "river cafe", "san domenico", "second avenue deli", "seryna", "shun lee west", "sign of the dove",
    "smith & wollensky", "tavern on the green", "uncle nick \s", "union square cafe", "virgil \s", "chin \s",
    "coyote cafe", "le montrachet", "palace court", "second street grille", "steak house", "tillerman", "abruzzi",
    "bacchanalia", "bone \s", "brasserie le coze", "buckhead diner", "ciboulette", "delectables", "georgia grille",
    "hedgerose heights inn", "heera of india", "indigo coastal grill", "la grotta", "mary mac \s tea room",
    "nikolai \s roof", "pano \s and paul \s", "cafe ritz-carlton buckhead", "dining room ritz-carlton buckhead",
    "restaurant ritz-carlton atlanta", "toulouse", "veni vidi vici", "alain rondelli", "aqua", "boulevard",
    "cafe claude", "campton place", "chez michel", "fleur de lys", "fringale", "hawthorne lane", "khan toke thai house",
    "la folie", "lulu", "masa \s", "mifune japan center kintetsu building", "plumpjack cafe", "postrio",
    "ritz-carlton restaurant and dining room", "rose pistola", "bolo", "il nido", "remi", "adriano \s ristorante",
    "barney greengrass", "beaurivage", "bistro garden", "border grill", "broadway deli", "ca \ brea", "ca \ del sol",
    "cafe pinot", "california pizza kitchen", "canter \s", "cava", "cha cha cha", "chan dara", "clearwater cafe",
    "dining room", "dive !", "drago", "drai \s", "dynasty room", "eclipse", "ed debevic \s", "el cholo", "gilliland \s",
    "gladstone \s", "hard rock cafe", "harry \s bar & american grill", "il fornaio cucina italiana",
    "jack sprat \s grill", "jackson \s farm", "jimmy \s", "joss", "le colonial", "le dome", "louise \s trattoria",
    "mon kee seafood restaurant", "morton \s", "nate \ n \ al \s", "nicola", "ocean avenue", "orleans",
    "pacific dining car", "paty \s", "pinot hollywood", "posto", "prego", "rj \s the rib joint", "remi",
    "restaurant horikawa", "roscoe \s house of chicken \ n \ waffles", "schatzi on main", "sofi", "swingers",
    "tavola calda", "the mandarin", "tommy tang \s", "tra di noi", "trader vic \s", "vida", "west beach cafe",
    "'20 mott", "' 9 jones street", "adrienne", "agrotikon", "aja", "alamo", "alley \s end", "ambassador grill",
    "american place", "anche vivolo", "arizona", "arturo \s", "au mandarin", "bar anise", "barbetta", "ben benson \s",
    "big cup", "billy \s", "boca chica", "boonthai", "bouterin", "brothers bar-b-q", "bruno",
    "bryant park grill roof restaurant and bp cafe", "c3", "ct", "cafe bianco", "cafe botanica", "cafe la fortuna",
    "cafe luxembourg", "cafe pierre", "cafe centro", "cafe fes", "caffe dante", "caffe dell \ artista", "caffe lure",
    "caffe reggio", "caffe roma", "caffe vivaldi", "caffe bondi ristorante", "capsouto freres", "captain \s table",
    "casa la femme", "cendrillon asian grill & marimba bar", "chez jacqueline", "chiam", "china grill", "cite",
    "coco pazzo", "columbus bakery", "corrado cafe", "cupcake cafe", "da nico", "dean & deluca", "diva", "dix et sept",
    "docks", "duane park cafe", "el teddy \s", "'em ily \s", "'em pire korea", "ernie \s", "evergreen cafe",
    "f. ille ponte ristorante", "felix", "ferrier", "fifty seven fifty seven", "film center cafe",
    "fiorello \s roman cafe", "firehouse", "first", "fishin eddie", "fleur de jour", "flowers", "follonico",
    "fraunces tavern", "french roast", "french roast cafe", "frico bar", "fujiyama mama", "gabriela \s", "gallagher \s",
    "gianni \s", "girafe", "global", "golden unicorn", "grand ticino", "halcyon", "hard rock cafe",
    "hi-life restaurant and lounge", "home", "hudson river club", "' i trulli", "il cortile", "inca grill", "indochine",
    "internet cafe", "ipanema", "jean lafitte", "jewel of india", "jimmy sung \s", "joe allen", "judson grill",
    "l \ absinthe", "l \ auberge", "l \ auberge du midi", "l \ udo", "la reserve", "lanza restaurant",
    "lattanzi ristorante", "layla", "le chantilly", "le colonial", "le gamin", "le jardin", "le madri", "le marais",
    "le perigord", "le select", "les halles", "lincoln tavern", "lola", "lucky strike", "mad fish", "main street",
    "mangia e bevi", "manhattan cafe", "manila garden", "marichu", "marquet patisserie", "match", "matthew \s",
    "mavalli palace", "milan cafe and coffee bar", "monkey bar", "montien", "morton \s", "motown cafe",
    "new york kom tang soot bul house", "new york noodletown", "newsbar", "odeon", "orso", "osteria al droge", "otabe",
    "pacifica", "palio", "pamir", "parioli romanissimo", "patria", "peacock alley", "pen & pencil", "penang soho",
    "persepolis", "planet hollywood", "pomaire", "popover cafe", "post house", "rain", "red tulip", "republic",
    "roettelle a. g", "rosa mexicano", "ruth \s chris", "s.p.q.r", "sal anthony \s", "sammy \s roumanian steak house",
    "san pietro", "sant ambroeus", "sarabeth \s kitchen", "sea grill", "serendipity", "seventh regiment mess and bar",
    "sfuzzi", "shaan", "sofia fabulous pizza", "spring street natural restaurant & bar", "stage deli", "stingray",
    "sweet \ n \ tart cafe", "' t salon", "tang pavillion", "tapika", "teresa \s", "terrace", "the coffee pot",
    "the savannah club", "trattoria dell \ arte", "triangolo", "tribeca grill", "trois jean", "tse yang",
    "turkish kitchen", "two two two", "veniero \s pasticceria", "verbena", "victor \s cafe", "vince & eddie \s", "vong",
    "water club", "west", "xunta", "zen palate", "zoe", "abbey", "aleck \s barbecue heaven", "annie \s thai castle",
    "anthonys", "atlanta fish market", "beesley \s of buckhead", "bertolini \s", "bistango", "cafe renaissance",
    "camille \s", "cassis", "city grill", "coco loco", "colonnade restaurant", "dante \s down the hatch buckhead",
    "dante \s down the hatch", "fat matt \s rib shack", "french quarter food shop", "holt bros. bar-b-q",
    "horseradish grill", "hsu \s gourmet", "imperial fez", "kamogawa", "la grotta at ravinia dunwoody rd.",
    "little szechuan", "lowcountry barbecue", "luna si", "mambo restaurante cubano", "mckinnon \s louisiane",
    "mi spia dunwoody rd.", "nickiemoto \s : a sushi bar", "palisades", "pleasant peasant", "pricci",
    "r.j. \s uptown kitchen & wine bar", "rib ranch", "sa tsu ki", "sato sushi and thai", "south city kitchen",
    "south of france", "stringer \s fish camp and oyster bar", "sundown cafe", "taste of new orleans", "tomtom",
    "antonio \s", "bally \s big kitchen", "bamboo garden", "battista \s hole in the wall", "bertolini \s",
    "binion \s coffee shop", "bistro", "broiler", "bugsy \s diner", "cafe michelle", "cafe roma", "capozzoli \s",
    "carnival world", "center stage plaza hotel", "circus circus", "'em press court", "feast", "golden nugget hotel",
    "golden steer", "lillie langtry \s", "mandarin court", "margarita \s mexican cantina", "mary \s diner", "mikado",
    "pamplemousse", "ralph \s diner", "the bacchanal", "venetian", "viva mercado \s", "yolie \s", "2223", "acquarello",
    "bardelli \s", "betelnut", "bistro roti", "bix", "bizou", "buca giovanni", "cafe adriano", "cafe marimba",
    "california culinary academy", "capp \s corner", "carta", "chevys", "cypress club", "des alpes", "faz",
    "fog city diner", "garden court", "gaylord \s", "grand cafe hotel monaco", "greens", "harbor village", "harris \'",
    "harry denton \s", "hayes street grill", "helmand", "hong kong flower lounge", "hong kong villa",
    "hyde street bistro", "il fornaio levi \s plaza", "izzy \s steak & chop house", "jack \s", "kabuto sushi",
    "katia \s", "kuleto \s", "kyo-ya . sheraton palace hotel", "l \ osteria del forno", "le central", "le soleil",
    "macarthur park", "manora", "maykadeh", "mccormick & kuleto \s", "millennium", "moose \s", "north india",
    "one market", "oritalia", "pacific pan pacific hotel", "palio d \ asti", "pane e vino", "pastis", "perry \s",
    "r & g lounge", "rubicon", "rumpus", "sanppo", "scala \s bistro", "south park cafe", "splendido embarcadero",
    "stars", "stars cafe", "stoyanof \s cafe", "straits cafe", "suppenkuche", "tadich grill", "the heights", "thepin",
    "ton kiang", "vertigo", "vivande porta via", "vivande ristorante", "world wrapps", "wu kong", "yank sing",
    "yaya cuisine", "yoyo tsumami bistro", "zarzuela", "zuni cafe & grill"
]

random_int_field = [0, 162, 213, 220, 206, 152, 184, 168, 176, 156, 187, 214, 167, 189, 186, 172, 196, 173, 186, 182,
                    174, 198, 149, 229, 162, 192, 184, 207, 175, 154, 188, 213, 195, 205, 204, 193, 197, 184, 164, 163,
                    172, 194, 169, 161, 165, 189, 187, 209, 192, 211, 166, 201, 173, 181, 170, 163, 190, 192, 198, 210,
                    199, 198, 195, 165, 180, 155, 185, 221, 183, 196, 185, 208, 177, 163, 216, 205, 201, 221, 196, 169,
                    181, 199, 238, 173, 161, 189, 208, 193, 174, 197, 202, 181, 182, 178, 165, 166, 193, 181, 173, 177,
                    201, 168, 154, 178, 184, 179, 159, 167, 172, 174, 188, 201, 205, 179, 164, 223, 203, 189, 178, 223,
                    167, 180, 164, 150, 193, 202, 180, 196, 188, 172, 192, 166, 186, 178, 151, 210, 174, 180, 168, 183,
                    170, 173, 181, 184, 206, 184, 202, 216, 159, 210, 179, 147, 196, 170, 170, 198, 180, 147, 177, 185,
                    177, 170, 199, 203, 193, 178, 163, 159, 172, 202, 170, 205, 170, 200, 179, 168, 178, 193, 162, 183,
                    181, 211, 166, 196, 202, 170, 202, 193, 180, 156, 235, 231, 204, 218, 223, 208, 181, 232, 171]

random_float_field = [73.84701702, 68.78190405, 74.11010539, 71.7309784, 69.88179586, 67.25301569, 68.78508125,
                      68.34851551, 67.01894966, 63.45649398, 71.19538228, 71.64080512, 64.76632913, 69.2830701,
                      69.24373223, 67.6456197, 72.41831663, 63.97432572, 69.6400599, 67.93600485, 67.91505019,
                      69.43943987, 66.14913196, 75.20597361, 67.89319634, 68.1440328, 69.08963143, 72.80084352,
                      67.42124228, 68.49641536, 68.61811055, 74.03380762, 71.52821604, 69.1801611, 69.57720237,
                      70.40092889, 69.07617117, 67.19352328, 65.80731565, 64.30418789, 67.97433623, 72.18942596,
                      65.27034552, 66.09017738, 67.51032152, 70.10478626, 68.25183644, 72.17270912, 69.17985762,
                      72.87036015, 64.78258298, 70.18354989, 68.49145025, 67.33083088, 66.99094408, 66.4995499,
                      68.35305665, 70.77445907, 71.21592367, 70.01336535, 71.40318222, 69.55200509, 73.81853456,
                      66.99688275, 71.41846589, 65.27930021, 68.27419147, 72.76536995, 68.0993798, 68.89670607,
                      69.28950996, 70.52322452, 69.66372523, 67.59526881, 72.50812038, 71.2529856, 71.80918689,
                      72.24516548, 66.51262766, 66.029034, 67.57715394, 68.2465686, 73.826127, 69.80246436, 65.95957778,
                      71.07901758, 66.59619654, 68.95153509, 68.24446179, 72.31682512, 71.81542045, 65.23704952,
                      70.64053009, 64.7319256, 67.10355118, 65.11748489, 71.70123402, 66.83287821, 66.47127526,
                      69.41152622, 70.05217747, 66.74360465, 66.27432912, 68.32844799, 70.0758882, 68.73298815,
                      67.55605126, 66.25363253, 69.18220268, 67.60910494, 69.29273802, 68.19068401, 71.6070858,
                      69.19685751, 67.26196098, 73.6851934, 69.53721501, 68.31155984, 67.73896347, 71.7057626,
                      63.6322646, 68.72119846, 66.94934165, 62.70698974, 72.25840892, 70.90865306, 67.6098436,
                      70.80155896, 69.30476905, 66.24289834, 67.49219298, 65.80624829, 71.44370566, 68.46440582,
                      63.9879246, 71.00189769, 68.13972419, 68.39540025, 68.09621975, 68.14059036, 68.86009031,
                      66.14885254, 66.20603205, 67.43212021, 69.47110603, 70.51585969, 71.33837604, 71.00194477,
                      66.20234771, 72.54330705, 67.47935176, 65.35041056, 70.84406242, 69.93847526, 64.73981548,
                      69.30840288, 68.83846286, 61.93732327, 68.59333554, 65.21857558, 64.33364811, 68.7489067,
                      72.4896554, 67.23393092, 67.26360484, 65.11850428, 66.26282004, 67.70167966, 65.53069597,
                      69.86896981, 68.48187536, 72.21396335, 68.17953269, 71.98120654, 66.06513673, 66.65616417,
                      67.5994242, 68.24594409, 64.80862144, 67.49221827, 68.18073071, 69.5533849, 66.40224967,
                      66.59215711, 71.93588658, 68.28704173, 69.9554512, 71.85112915, 65.75549864, 67.03185208,
                      76.70983486, 72.57112137, 69.7288049, 72.799224, 72.53935407, 72.29474338, 67.25332482,
                      75.94446038, 66.31623192]


class ItunesAmazonAlter:

    def __init__(self, mode, attr_sep='ATTR'):
        func_map = {'SFF': self._sff,
                    'DRP': self._drp,
                    'MIS': self._mis,
                    'TYP': self._typ,
                    'EXT': self._ext}

        self.mode = mode
        self.attr_sep = attr_sep
        self.alter_func = func_map[mode]
        self.col_size = 8
        self.col_types = ['id', 'id', 'id', 'text', 'price', 'text', 'time', 'date']
        self.permutation = 1
        self.key_col = [0, 1, 2]
        rand_lst = [i for i in range(self.col_size)]
        random.shuffle(rand_lst)

        if mode == 'SFF':
            # 0:left, 1:right
            self.sff_side = random.randint(0, 1)
            self.sff_order = self._generate_shuffle_order(self.col_size)[0:self.permutation]
        elif mode == 'DRP':
            self.drp_side = random.randint(0, 1)
            self.drp_cols = sorted(rand_lst[0:self.permutation], reverse=True)
        elif mode == 'MIS':
            rand_lst = [i for i in range(3, self.col_size)]
            random.shuffle(rand_lst)
            self.mis_side = random.randint(0, 1)
            self.mis_cols = rand_lst[0:self.permutation]
        elif mode == 'TYP':
            self.typ_side = random.randint(0, 1)
            rand_lst = [4, 6, 7]
            random.shuffle(rand_lst)
            self.typ_cols = rand_lst[0:self.permutation]
        elif mode == 'EXT':
            typ_lst = ['text', 'int', 'float']
            random.shuffle(typ_lst)
            self.ext_type = typ_lst[0:self.permutation]
            self.ext_side = random.randint(0, 1)
            self.ext_options = {
                'text': random_text_field,
                'int': random_int_field,
                'float': random_float_field
            }
        else:
            raise ValueError(f'mode:{mode} is an unknown Alter mode for ItunesAmazon.')

    @staticmethod
    def _generate_shuffle_order(size):
        from_list = [i for i in range(size)]
        to_list = [i for i in range(size)]
        random.shuffle(to_list)
        return [(i, j) for i, j in zip(from_list, to_list)]

    def _drp(self, left, right):
        """
        DRP: Dropping a non-key column (This is different from missing values).
        We randomly remove one or more non-key columns. #  from both side? is the dropped column same for all data points?
        :return:
        """
        for d in self.drp_cols:
            if self.drp_side == 0:
                left.pop(d)
            else:
                right.pop(d)
        return left, right

    def _sff(self, left, right):
        """
        SFF: Shuffling columns. For a pair of entities, we shuffle only one of them.
        :return:
        """
        for orders in self.sff_order:
            from_idx, to_idx = orders
            if self.sff_side == 0:
                left[to_idx], left[from_idx] = left[from_idx], left[to_idx]
            else:
                right[to_idx], right[from_idx] = right[from_idx], right[to_idx]

        return left, right

    def _mis(self, left, right):
        """
        MIS: Replace a non-key column with missing value.
        :return:
        """

        for col in self.mis_cols:
            if self.mis_side == 0:
                left[col] = '<UNK>'
            else:
                right[col] = '<UNK>'

        return left, right

    def _typ(self, left, right):
        """
        TYP: For numerical columns, convert their types to different formats. For instance,
        for price, we can convert it to string or add a dollar sign or divide it by 1000
        like 9000->9K.
        :return:
        """
        for col in self.typ_cols:
            col_type = self.col_types[col]
            if col_type == 'price':
                # itunes default price format: $ 1.29
                if self.typ_side == 0:
                    left[col] = left[col].replace("$ ", "")
                else:
                    right[col] = right[col].replace("$ ", "")
            elif col_type == 'time':
                # most rows: 0X:YZ some rows: 0X:YZ:MN
                if len(left[col]) == 5 and self.typ_side == 0:
                    left[col] = left[col] + ":00"
                if len(right[col]) == 5 and self.typ_side == 1:
                    right[col] = right[col] + ":00"
            elif col_type == 'date':
                # tableB: February 9 , 2011 tableA: 9-Jan-15
                if self.typ_side == 0:
                    try:
                        left[col] = str(parser.parse(left[col]))
                    except:
                        return left, right
                else:
                    try:
                        right[col] = str(parser.parse(right[col]))
                    except:
                        return left, right

        return left, right

    def _ext(self, left, right):
        """
        EXT: Adding one or more irrelevant columns. The new columns can be of type integer,
        floating-point, or string. For text columns, I suggest using columns from other
        datasets rather than random words. We can have multiple test data for this.
        :return:
        """
        for ext in self.ext_type:
            if self.ext_side == 0:
                left.insert(random.randint(0, len(left)), random.choice(self.ext_options[ext]))
            else:
                right.insert(random.randint(0, len(right)), random.choice(self.ext_options[ext]))

        return left, right

    def alter(self, left, left_full, right, right_full):
        left, right = self.alter_func(left, right)
        left_full = f' {self.attr_sep} '.join([str(x) for x in left]).strip()
        right_full = f' {self.attr_sep} '.join([str(x) for x in right]).strip()
        return left, left_full, right, right_full
