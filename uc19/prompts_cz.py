"""UC19: 200 Czech image requests (150 běžných + 50 záludných) and 15 few-shot examples per dialect.

Source: written for this benchmark by the agent (Claude, 2026-10-29), CC0. No user / Ol1nLLM log data.
All SFW (adult figures only, no nudity/violence/real people); checked by common.sfw_hits in tests.

Eval line:  category | style | czech request || English gist of the INTENDED meaning || must_tags || must_not
  category: plain | idiom (literal translation is wrong) | realia (Czech-specific thing) | negation |
            diacritics (no diacritics / colloquial / typo)
  ref_en    = independent reference for PickScore / SigLIP / VLM judge (never shown to the translator)
  must_tags = WD14 tags that should be detected; must_not = WD14 tags that must NOT appear (negation, literal traps)
`python3 prompts_cz.py --out prompts/cz_prompts.jsonl --fewshot prompts/fewshot.jsonl`
"""
import argparse, json, sys

SOURCE = "written by Claude (agent) for UC19, 2026-10-29"
LICENSE = "CC0-1.0"

EVAL = """
plain | photo_portrait | Portrét starší ženy s krátkými šedými vlasy a kulatými brýlemi, měkké světlo z okna || portrait of an elderly woman with short grey hair and round glasses, soft window light || grey hair, short hair, glasses, portrait ||
plain | photo_portrait | Vousatý rybář v pletené čepici v přístavu, zamračená obloha || bearded fisherman in a knitted cap at a harbor, overcast sky || beard, beanie, harbor, overcast ||
plain | photo_portrait | Mladý muž s kudrnatými vlasy se směje na střeše domu při západu slunce || young man with curly hair laughing on a rooftop at sunset || curly hair, laughing, rooftop, sunset ||
plain | photo_portrait | Kuchař v bílém rondonu servíruje dezert v rušné kuchyni || chef in a white chef jacket plating a dessert in a busy kitchen || chef, chef uniform, kitchen, plate ||
plain | photo_portrait | Žena s dlouhými zrzavými vlasy v zeleném kabátě v podzimním parku || woman with long red hair in a green coat in an autumn park || long hair, red hair, green coat, autumn leaves ||
plain | photo_portrait | Černobílá fotka saxofonisty na pódiu v zakouřeném klubu || black and white photo of a saxophonist on a smoky club stage || saxophone, monochrome, stage, smoke ||
plain | photo_portrait | Hrnčířka u hrnčířského kruhu v dílně zalité sluncem || woman potter at a pottery wheel in a sunlit workshop || pottery, apron, workshop, sunlight ||
plain | photo_portrait | Starý manželský pár sedí na lavičce v zahradě a drží se za ruce || elderly couple sitting on a garden bench holding hands || old man, old woman, bench, holding hands ||
plain | photo_portrait | Astronautka v bílém skafandru drží helmu v hangáru || female astronaut in a white spacesuit holding her helmet in a hangar || astronaut, spacesuit, holding helmet ||
plain | photo_portrait | Cyklista ve žluté bundě projíždí v noci mokrou ulicí s neony || cyclist in a yellow jacket riding a wet neon-lit street at night || bicycle, yellow jacket, night, neon lights, rain ||
plain | photo_portrait | Houslistka v dlouhých tmavých šatech hraje na koncertním pódiu || female violinist in a long dark dress playing on a concert stage || violin, black dress, stage, spotlight ||
plain | photo_portrait | Kovář buší kladivem do rozžhaveného železa, létají jiskry || blacksmith hammering glowing iron, flying sparks || blacksmith, hammer, anvil, sparks ||
plain | photo_portrait | Zahradnice ve slaměném klobouku s košíkem rajčat ve skleníku || gardener woman in a straw hat with a basket of tomatoes in a greenhouse || straw hat, basket, tomato, greenhouse ||
plain | photo_portrait | Vědkyně v laboratorním plášti se dívá do mikroskopu || female scientist in a lab coat looking into a microscope || lab coat, microscope, laboratory ||
plain | photo_portrait | Hasič v plné výstroji se opírá o červené auto, helmu v podpaží || firefighter in full gear leaning on a red truck, helmet under his arm || firefighter, helmet, fire truck ||
plain | photo_landscape | Mlhavý borový les za úsvitu, sluneční paprsky mezi stromy || misty pine forest at dawn with sun rays between trees || forest, pine tree, fog, light rays, no humans ||
plain | photo_landscape | Tyrkysové horské jezero obklopené zasněženými štíty || turquoise mountain lake surrounded by snowy peaks || lake, mountain, snow, reflection ||
plain | photo_landscape | Maják na skalnatém útesu během bouře, vlny se tříští o skály || lighthouse on a rocky cliff during a storm, crashing waves || lighthouse, cliff, waves, storm ||
plain | photo_landscape | Písečné duny při západu slunce s dlouhými stíny || sand dunes at sunset with long shadows || desert, dune, sunset, shadow ||
plain | photo_landscape | Polární záře nad zamrzlým jezerem a malou chatkou || northern lights over a frozen lake and a small cabin || aurora, frozen lake, cabin, night sky ||
plain | photo_landscape | Úzká dlážděná ulička se truhlíky s květinami za oknem || narrow cobblestone alley with flower boxes on the windows || alley, cobblestone, flower, window ||
plain | photo_landscape | Levandulové pole v řádcích, na konci kamenný statek || lavender field in rows leading to a stone farmhouse || lavender, flower field, house ||
plain | photo_landscape | Vodopád v zelené džungli s kapradinami a mechem || waterfall in a green jungle with ferns and moss || waterfall, jungle, fern, moss ||
plain | photo_landscape | Zasněžená vesnice s kouřícími komíny za soumraku || snowy village with smoking chimneys at dusk || village, snow, chimney, smoke, dusk ||
plain | photo_landscape | Noční panorama města přes řeku, světelné stopy aut na mostě || night city skyline across a river with car light trails on a bridge || cityscape, night, river, bridge, light trail ||
plain | photo_landscape | Pole slunečnic pod modrou letní oblohou || sunflower field under a blue summer sky || sunflower, flower field, blue sky ||
plain | photo_landscape | Japonská zahrada s červeným dřevěným mostkem a jezírkem s kapry || japanese garden with a red wooden bridge and a koi pond || japanese garden, red bridge, pond, koi ||
plain | photo_landscape | Kaňon s červenými skalami a klikatící se řekou || canyon with red rock walls and a winding river || canyon, rock, river ||
plain | photo_landscape | Bambusový les s cestičkou v mlze || misty bamboo forest path || bamboo forest, path, fog ||
plain | photo_landscape | Přístav s rybářskými loďkami při východu slunce, nad nimi racci || harbor with fishing boats at sunrise and seagulls above || harbor, boat, sunrise, seagull ||
plain | product_food | Miska ramenu s vajíčkem natvrdo, jarní cibulkou a řasou nori || bowl of ramen with a boiled egg, spring onion and nori || ramen, bowl, egg, nori ||
plain | product_food | Komín nadýchaných palačinek s borůvkami a javorovým sirupem || stack of fluffy pancakes with blueberries and maple syrup || pancake, stack, blueberry, syrup ||
plain | product_food | Ledová káva ve sklenici na mramorové desce, minimalistická produktová fotka || iced coffee in a glass on a marble counter, minimalist product photo || iced coffee, ice cube, drinking glass, marble ||
plain | product_food | Starý kinofilmový fotoaparát na dřevěném stole vedle zápisníku || vintage film camera on a wooden desk next to a notebook || camera, desk, notebook ||
plain | product_food | Kousek jahodového dortu se šlehačkou na bílém talíři || slice of strawberry cake with whipped cream on a white plate || cake slice, strawberry, whipped cream, plate ||
plain | product_food | Bochník kváskového chleba na lněné utěrce s nožem || loaf of sourdough bread on a linen cloth with a knife || bread, knife, cloth ||
plain | product_food | Neapolská pizza s bazalkou těsně po vytažení z pece || neapolitan pizza with basil fresh out of a wood oven || pizza, basil, oven ||
plain | product_food | Konvička na čaj a dva šálky na bambusovém podnose, stoupá pára || teapot and two cups on a bamboo tray with rising steam || teapot, cup, tray, steam ||
plain | product_food | Sushi rolky a nigiri na břidlicové desce s hůlkami || sushi rolls and nigiri on a slate board with chopsticks || sushi, chopsticks ||
plain | product_food | Hrnek horké čokolády s marshmallow vedle pleteného šálu || mug of hot chocolate with marshmallows next to a knitted scarf || mug, hot chocolate, marshmallow, scarf ||
plain | product_food | Kytice tulipánů ve skleněné váze na kuchyňském stole || bouquet of tulips in a glass vase on a kitchen table || tulip, bouquet, vase, table ||
plain | product_food | Burger s roztaveným sýrem a hranolky na dřevěném prkénku || burger with melted cheese and fries on a wooden board || hamburger, cheese, french fries ||
plain | product_food | Sklenice medu s dřevěnou naběračkou, med stéká || jar of honey with a wooden dipper, dripping honey || honey, jar ||
plain | product_food | Mechanická klávesnice s barevnými klávesami na tmavém stole || mechanical keyboard with colorful keycaps on a dark desk || keyboard (computer), desk ||
plain | product_food | Košík čerstvé zeleniny: mrkev, pórek a ředkvičky || basket of fresh vegetables: carrots, leeks and radishes || vegetable, carrot, basket ||
plain | animals | Liška sedí v čerstvém sněhu v zimním lese || red fox sitting in fresh snow in a winter forest || fox, snow, forest, sitting ||
plain | animals | Štěně zlatého retrívra běží přes rozkvetlou louku || golden retriever puppy running across a flowering meadow || dog, golden retriever, running, flower ||
plain | animals | Sova pálená sedí na sloupku plotu za soumraku || barn owl perched on a fence post at dusk || owl, fence, dusk ||
plain | animals | Zrzavá kočka spí na slunném parapetu vedle květináče || ginger cat sleeping on a sunny windowsill next to a flowerpot || cat, orange cat, sleeping, windowsill ||
plain | animals | Stádo slonů jde přes savanu při západu slunce || herd of elephants walking across the savanna at sunset || elephant, savannah, sunset ||
plain | animals | Kolibřík u červeného květu, makro || hummingbird next to a red flower, macro || hummingbird, red flower, flying ||
plain | animals | Mořská želva plave nad korálovým útesem || sea turtle swimming over a coral reef || sea turtle, underwater, coral reef ||
plain | animals | Lev odpočívá na skále v teplém večerním světle || lion resting on a rock in warm evening light || lion, rock, sunset ||
plain | animals | Dva tučňáci stojí na ledu, za nimi oceán || two penguins standing on ice with the ocean behind them || penguin, ice, ocean ||
plain | animals | Veverka drží lískový oříšek na mechové větvi || squirrel holding a hazelnut on a mossy branch || squirrel, nut, branch, moss ||
plain | animals | Kůň cválá po pláži při východu slunce || horse galloping on a beach at sunrise || horse, galloping, beach, sunrise ||
plain | animals | Sněžný levhart na skalnatém svahu v sněžení || snow leopard on a rocky slope in falling snow || snow leopard, rock, snowing ||
plain | animals | Ježek leze po spadaném listí na podzimní zahradě || hedgehog walking over fallen leaves in an autumn garden || hedgehog, autumn leaves, garden ||
plain | animals | Plameňák stojí na jedné noze v mělké laguně || flamingo standing on one leg in a shallow lagoon || flamingo, standing on one leg, water ||
plain | animals | Panda červená šplhá po větvi stromu || red panda climbing a tree branch || red panda, climbing, branch ||
plain | anime_character | Anime ilustrace mladé ženy se stříbrnými vlasy a modrýma očima v bílém kabátě, padá sníh || anime illustration of a young woman with silver hair and blue eyes in a white coat in falling snow || silver hair, blue eyes, white coat, snowing ||
plain | anime_character | Anime rytířka s krátkými blond vlasy ve stříbrné zbroji drží meč || anime knight woman with short blonde hair in silver armor holding a sword || knight, blonde hair, short hair, armor, sword ||
plain | anime_character | Anime servírka s culíkem nese tác s kávou v kavárně || anime waitress with a ponytail carrying a tray of coffee in a cafe || waitress, ponytail, holding tray, cafe ||
plain | anime_character | Anime čarodějka s fialovými vlasy sesílá modré kouzlo v knihovně || anime mage woman with purple hair casting a blue spell in a library || purple hair, magic, library, robe ||
plain | anime_character | Anime muž v obleku s brýlemi čte dokumenty v kanceláři v noci || anime man in a suit with glasses reading documents in an office at night || suit, glasses, office, night ||
plain | anime_character | Anime lukostřelkyně se zelenými vlasy míří lukem v lese || anime archer woman with green hair aiming a bow in a forest || green hair, bow (weapon), aiming, forest ||
plain | anime_character | Anime samuraj s dlouhým culíkem v tmavém kimonu, padají květy sakury || anime samurai with a long ponytail in a dark kimono, falling cherry blossoms || samurai, ponytail, kimono, cherry blossoms ||
plain | anime_character | Anime detektiv v baloňáku a klobouku s lupou v mlhavé ulici || anime detective in a trench coat and hat with a magnifying glass in a foggy street || detective, trench coat, magnifying glass, fog ||
plain | anime_character | Anime malířka s krátkými hnědými vlasy maluje na plátno, má barvu na tváři || anime woman painter with short brown hair painting on a canvas, paint on her cheek || brown hair, short hair, paintbrush, paint on face ||
plain | anime_character | Anime žena v modrém kimonu s papírovým lampionem na letní slavnosti v noci || anime woman in a blue kimono with a paper lantern at a summer festival at night || kimono, paper lantern, summer festival, night ||
plain | anime_character | Anime mechanička s brýlemi na čele opravuje motorku v garáži || anime mechanic woman with goggles on her forehead repairing a motorcycle in a garage || goggles on head, motorcycle, garage, wrench ||
plain | anime_character | Anime muž hraje na akustickou kytaru na lavičce v parku na podzim || anime man playing an acoustic guitar on a park bench in autumn || acoustic guitar, bench, park, autumn ||
plain | anime_character | Anime čarodějnice letí na koštěti nad městem za soumraku s černou kočkou || anime witch flying on a broom over a town at dusk with a black cat || witch, witch hat, riding broom, black cat ||
plain | anime_character | Anime knihovnice s brýlemi a dlouhým copem šplhá po žebříku pro knihu || anime librarian with glasses and a long braid climbing a ladder for a book || glasses, braid, ladder, library ||
plain | anime_character | Anime kapitánka s čepicí u kormidla lodi, rozbouřené moře || anime woman ship captain with a cap at the helm, rough sea || captain, ship's wheel, ship, ocean ||
plain | anime_scenery | Anime pozadí venkovské vlakové zastávky v létě s kupovitými mraky || anime background of a rural train station in summer with cumulus clouds || train station, summer, cloud, no humans ||
plain | anime_scenery | Anime scenérie dřevěného stánku s ramenem v noci s lampiony || anime scenery of a wooden ramen stall at night with lanterns || food stand, lantern, night ||
plain | anime_scenery | Anime létající ostrovy s vodopády na modré obloze || anime floating islands with waterfalls in a blue sky || floating island, waterfall, blue sky ||
plain | anime_scenery | Anime prázdná třída při západu slunce, oranžové světlo okny || anime empty classroom at sunset with orange light through the windows || classroom, desk, sunset, window ||
plain | anime_scenery | Anime brána svatyně v lese s rozkvetlými sakurami a kamennými lucernami || anime shrine gate in a forest with cherry blossoms and stone lanterns || torii, cherry blossoms, stone lantern, forest ||
plain | anime_scenery | Anime útulný pokoj s rostlinami, lampičkou a deštivým oknem v noci || anime cozy room with plants, a desk lamp and a rainy window at night || plant, desk lamp, window, rain, night ||
plain | anime_scenery | Anime přímořské městečko s příkrými schody a oranžovými střechami || anime seaside town with steep stairs and orange roofs || town, stairs, ocean, roof ||
plain | anime_scenery | Anime hvězdná obloha nad kopcem s osamělým stromem, mléčná dráha || anime starry sky over a hill with a lone tree, milky way || starry sky, milky way, hill, tree ||
plain | anime_scenery | Anime opuštěný zábavní park s rezavým ruským kolem zarostlým květinami || anime abandoned amusement park with a rusty ferris wheel overgrown with flowers || ferris wheel, amusement park, overgrown, flower ||
plain | anime_scenery | Anime vlak jede po mostě přes moře, jasná obloha || anime train crossing a bridge over the sea, clear sky || train, bridge, ocean, blue sky ||
plain | anime_scenery | Anime kouzelná knihovna s nekonečnými regály a létajícími knihami || anime magical library with endless shelves and floating books || library, bookshelf, floating object ||
plain | anime_scenery | Anime kyberpunková ulička s hologramy a kalužemi s růžovým odrazem || anime cyberpunk alley with holograms and puddles reflecting pink light || cyberpunk, hologram, puddle, neon lights ||
plain | anime_scenery | Anime hrad na útesu nad mraky při východu slunce || anime castle on a cliff above the clouds at sunrise || castle, cliff, above clouds, sunrise ||
plain | anime_scenery | Anime podvodní město se svítícími kopulemi a rybami || anime underwater city with glowing domes and fish || underwater, city, dome, fish ||
plain | anime_scenery | Anime obrovský strom s domkem v koruně a lampiony za soumraku || anime giant tree with a treehouse and lanterns at dusk || tree, treehouse, lantern, dusk ||
plain | painting | Olejomalba rozbouřeného moře s plachetnicí || oil painting of a stormy sea with a sailing ship || oil painting (medium), sailboat, waves, storm ||
plain | painting | Akvarel klidné přístavní vesnice s pastelovými domky || watercolor of a quiet harbor village with pastel houses || watercolor (medium), harbor, house, pastel colors ||
plain | painting | Impresionistický obraz ženy se slunečníkem v makovém poli || impressionist painting of a woman with a parasol in a poppy field || parasol, flower field, impressionism ||
plain | painting | Japonský dřevořez velké vlny s horou Fudži v pozadí || japanese woodblock print of a great wave with mount fuji in the background || ukiyo-e, waves, mount fuji ||
plain | painting | Zátiší s jablky, měděným džbánem a chlebem ve stylu holandských mistrů || still life with apples, a copper jug and bread in the style of dutch masters || still life, apple, jug, bread ||
plain | painting | Akvarelová liška v zimním lese, jemné přechody barev || watercolor fox in a winter forest, soft washes || fox, watercolor (medium), snow, forest ||
plain | painting | Čínská tušová malba mlžných hor s loďkou a borovicemi || chinese ink wash painting of misty mountains with a small boat and pines || ink wash painting, mountain, fog, boat ||
plain | painting | Pastelová kresba baletky, která si zavazuje špičky || pastel drawing of a ballerina tying her pointe shoes || ballerina, pastel (medium), ballet slippers, sitting ||
plain | painting | Olejomalba slunečnicového pole pod vířící hvězdnou oblohou || oil painting of a sunflower field under a swirling starry sky || sunflower, starry sky, oil painting (medium) ||
plain | painting | Secesní plakát ženy s vlajícími vlasy obklopené květinami || art nouveau poster of a woman with flowing hair surrounded by flowers || art nouveau, long hair, flower ||
plain | painting | Uhlová kresba lví hlavy na strukturovaném papíře || charcoal drawing of a lion head on textured paper || lion, monochrome, sketch ||
plain | painting | Akvarel větvičky třešňových květů s ptáčky || watercolor of a cherry blossom branch with small birds || cherry blossoms, branch, bird, watercolor (medium) ||
plain | painting | Olejomalba zasněžené vesnice s bruslaři na zamrzlém rybníku || oil painting of a snowy village with skaters on a frozen pond || snow, village, ice skating ||
plain | painting | Mozaika páva se zlatým pozadím v byzantském stylu || byzantine mosaic of a peacock with a gold background || peacock, mosaic, gold background ||
plain | painting | Olejomalba tiché knihovny s lampou a starými knihami || oil painting of a quiet library with a lamp and old books || library, lamp, book, oil painting (medium) ||
plain | render_3d | 3D render roztomilého izometrického domečku na létajícím ostrůvku, pastelové barvy || 3d render of a cute isometric tiny house on a floating island, pastel colors || 3d, isometric, house, floating island ||
plain | render_3d | 3D model z plastelíny: usměvavý kaktus v květináči || 3d clay render of a smiling cactus in a pot || 3d, cactus, potted plant, smile ||
plain | render_3d | Low poly krajina s horami, jezerem a borovicemi při západu slunce || low poly landscape with mountains, a lake and pines at sunset || low poly, mountain, lake, sunset ||
plain | render_3d | 3D render futuristického sporťáku v tmavém studiu s odlesky || 3d render of a futuristic sports car in a dark studio with reflections || 3d, sports car, reflection ||
plain | render_3d | 3D postavička přátelského robota s velkýma kulatýma očima, mává || 3d character of a friendly robot with big round eyes waving || 3d, robot, waving ||
plain | render_3d | 3D render skleněné koule s malým lesem uvnitř || 3d render of a glass sphere with a tiny forest inside || 3d, sphere, terrarium, tree ||
plain | render_3d | 3D vesnička z hub, která v noci svítí || 3d mushroom village glowing at night || 3d, mushroom house, glowing, night ||
plain | render_3d | 3D křeček astronaut ve skafandru se vznáší mezi planetami || 3d hamster astronaut in a spacesuit floating among planets || 3d, hamster, spacesuit, planet ||
plain | render_3d | 3D geoda rozříznutá napůl s fialovým ametystem || 3d geode cut in half with purple amethyst || 3d, crystal, amethyst, purple theme ||
plain | render_3d | 3D liška z plastelíny drží lucernu v zasněženém lese || 3d clay fox holding a lantern in a snowy forest || 3d, fox, lantern, snow ||
plain | render_3d | 3D vizualizace minimalistického obýváku s velkými okny || 3d visualization of a minimalist living room with large windows || 3d, living room, window, couch ||
plain | render_3d | 3D steampunková mechanická sova s mosaznými ozubenými koly || 3d steampunk mechanical owl with brass gears || 3d, steampunk, owl, gears ||
plain | render_3d | 3D tučňák kuchař s naběračkou v pixarovském stylu || 3d penguin chef holding a ladle, pixar style || 3d, penguin, chef hat, ladle ||
plain | render_3d | 3D sci-fi chodba se svítícími panely a mlhou || 3d sci-fi corridor with glowing panels and fog || 3d, corridor, science fiction, fog ||
plain | render_3d | 3D bonsaj na kamenném podstavci, studiové světlo || 3d bonsai tree on a stone pedestal, studio light || 3d, bonsai, stone ||
plain | graphic | Pixel art chaloupky na lesní mýtině s táborákem || pixel art of a cottage in a forest clearing with a campfire || pixel art, cottage, campfire, forest ||
plain | graphic | Plochá vektorová ilustrace hor se sluncem a ptáky, omezená paleta || flat vector illustration of mountains with sun and birds, limited palette || flat color, mountain, sun, bird ||
plain | graphic | Perokresba stočené spící kočky, čisté černé linky na bílé || line art of a curled up sleeping cat, clean black lines on white || lineart, cat, sleeping, white background ||
plain | graphic | Komiksový panel superhrdinky letící nad městem, rastrové stínování || comic panel of a superhero woman flying over a city, halftone shading || superhero, flying, city, halftone ||
plain | graphic | Pixel art vesmírné lodi letící kolem planety s prstenci || pixel art of a spaceship flying past a ringed planet || pixel art, spacecraft, planetary ring ||
plain | graphic | Samolepka veselého avokáda s bílým okrajem, kawaii styl || sticker of a happy avocado with a white border, kawaii style || sticker, avocado, white border, smile ||
plain | graphic | Minimalistický geometrický plakát lišky z trojúhelníků || minimalist geometric poster of a fox made of triangles || fox, geometric, triangle ||
plain | graphic | Pixel art rytíře bojujícího se slizem v kobce, retro RPG || pixel art of a knight fighting a slime in a dungeon, retro RPG || pixel art, knight, slime (creature), dungeon ||
plain | graphic | Sada plochých ikon počasí: slunce, mrak, déšť, sníh, blesk || set of flat weather icons: sun, cloud, rain, snow, lightning || icon, sun, cloud, lightning ||
plain | graphic | Omalovánka mandaly s květinami, černá na bílé || coloring page of a flower mandala, black on white || lineart, mandala, flower, monochrome ||
plain | graphic | Komiksový pes detektiv v baloňáku s výraznými obrysy || comic dog detective in a trench coat with bold outlines || dog, detective, trench coat ||
plain | graphic | Pixel art portrét čaroděje s dlouhým bílým plnovousem a hvězdnou čepicí || pixel art portrait of a wizard with a long white beard and a starry hat || pixel art, wizard, white beard, wizard hat ||
plain | graphic | Ilustrace ve stylu vitráže: sova s tlustými černými linkami || stained glass style owl with bold black lines || owl, stained glass ||
plain | graphic | Technický výkres parní lokomotivy jako modrotisk, bílé linky na modré || blueprint technical drawing of a steam locomotive, white lines on blue || blueprint, locomotive, blue background ||
plain | graphic | Pixel art hradu na kopci při západu slunce, titulní obrazovka retro hry || pixel art castle on a hill at sunset, retro game title screen || pixel art, castle, hill, sunset ||
plain | fantasy_scifi | Koncept art obřího pradávného stromu-města s mosty a svítícími okny || concept art of a giant ancient tree city with bridges and glowing windows || giant tree, city, bridge, glowing ||
plain | fantasy_scifi | Drak letí nad zasněženým pohořím za úsvitu || dragon flying over a snowy mountain range at dawn || dragon, flying, mountain, snow ||
plain | fantasy_scifi | Vesmírná stanice na oběžné dráze plynného obra || space station orbiting a gas giant || space station, planet, space ||
plain | fantasy_scifi | Steampunková vzducholoď kotví v létajícím přístavu nad mraky || steampunk airship docked at a floating harbor above the clouds || airship, steampunk, above clouds ||
plain | fantasy_scifi | Elfka hraničářka s dlouhými blond vlasy a zeleným pláštěm v kouzelném lese || elf ranger woman with long blonde hair and a green cloak in an enchanted forest || elf, pointy ears, blonde hair, green cloak, forest ||
plain | fantasy_scifi | Neonové kyberpunkové velkoměsto v noci s létajícími auty || neon cyberpunk megacity at night with flying cars || cyberpunk, city, night, flying car ||
plain | fantasy_scifi | Karavana velbloudů míjí starověké ruiny pod dvěma slunci || camel caravan passing ancient ruins under two suns || camel, desert, ruins ||
plain | fantasy_scifi | Čarodějova věž na útesu během bouřky s blesky || wizard tower on a cliff during a thunderstorm with lightning || tower, cliff, storm, lightning ||
plain | fantasy_scifi | Bojový robot stojí v deštivém průmyslovém areálu pod reflektory || mech robot standing in a rainy industrial yard under floodlights || mecha, robot, rain, industrial ||
plain | fantasy_scifi | Trpaslík kovář kuje svítící meč v horské síni || dwarf blacksmith forging a glowing sword in a mountain hall || dwarf, beard, anvil, sword, glowing ||
plain | fantasy_scifi | Fénix stoupá z plamenů, zlatá a červená pera || phoenix rising from flames with golden and red feathers || phoenix, fire, wings ||
plain | fantasy_scifi | Krystalová jeskyně se svítícími modrými houbami a podzemním jezerem || crystal cave with glowing blue mushrooms and an underground lake || cave, crystal, mushroom, glowing ||
plain | fantasy_scifi | Vikingská loď pluje fjordem pod polární září || viking longship sailing a fjord under the northern lights || ship, fjord, aurora ||
plain | fantasy_scifi | Obří želva nese na krunýři malou vesnici přes oceán || giant turtle carrying a small village on its shell across the ocean || turtle, village, ocean ||
plain | fantasy_scifi | Obří robot spí v poli květin, zarostlý mechem || giant robot sleeping in a field of flowers, overgrown with moss || robot, flower field, moss, overgrown ||
idiom | photo_landscape | Městská ulice, kde leje jako z konve || city street in a heavy downpour || rain, street, city || watering can
idiom | photo_portrait | Rybář s očima na stopkách napjatě sleduje splávek || fisherman intently staring at his fishing float || fishing rod, water, fishing ||
idiom | photo_portrait | Kuchař, který má plné zuby práce, v přeplněné kuchyni || chef who is fed up and overwhelmed with work in a crowded kitchen || chef, kitchen || teeth
idiom | photo_portrait | Muž s hlavou v oblacích sedí zasněně na lavičce v parku || daydreaming man sitting on a park bench, lost in thought || bench, park, sitting || cloud
idiom | photo_portrait | Úředník, který má máslo na hlavě, se nervózně ošívá u stolu || office clerk looking guilty and nervous at his desk || office, desk, necktie || butter
idiom | photo_landscape | Vesnice, kde chcípl pes: prázdná náves za šera || a deserted village square at dusk where nothing happens || village, dusk, no humans || dog
idiom | animals | Kočka, která je tu jako ryba ve vodě, leží spokojeně na gauči || a cat lying contentedly on a couch, perfectly at home || cat, couch, lying || fish
idiom | photo_portrait | Žena, která chodí kolem horké kaše, váhá před dveřmi kanceláře || a woman hesitating in front of an office door, avoiding a difficult conversation || door, hallway || bowl
idiom | photo_portrait | Turista, kterému spadl kámen ze srdce, se usmívá na vrcholu hory || a relieved hiker smiling on a mountain summit || mountain, smile, backpack || rock
idiom | fantasy_scifi | Hrad, kde dávají lišky dobrou noc, uprostřed hlubokého lesa || a remote castle in the middle of a deep forest, far from everything || castle, forest || fox
idiom | photo_portrait | Starý pán, který má zlaté ruce, opravuje hodiny v dílně || a skilled old craftsman repairing a clock in a workshop || old man, clock, workshop || gold
idiom | photo_portrait | Prodavač, který dělá z komára velblouda, rozhazuje rukama na trhu || a market vendor making a big fuss, waving his arms at a market stall || market stall || camel
idiom | photo_portrait | Programátor, který má nervy na pochodu, sedí před třemi monitory || a stressed programmer sitting in front of three monitors || monitor, computer, sitting ||
idiom | animals | Pes, který se tváří jako by mu uletěly včely, sedí na zápraží || a dog sitting on a porch looking sad and lost || dog, sitting, porch || bee
idiom | anime_character | Anime mladá žena, která je úplně v sedmém nebi, točí se v poli květin || anime young woman overjoyed, spinning in a flower field || flower field, smile, spinning ||
realia | photo_landscape | Karlův most v Praze za mlhavého rána se sochami svatých || charles bridge in prague on a misty morning with statues of saints || bridge, statue, fog, city ||
realia | photo_landscape | Staroměstský orloj zblízka, zlaté ciferníky || the prague astronomical clock close-up with golden dials || clock, tower, gold ||
realia | photo_landscape | Sněžka v zimě, chata na vrcholu a zasněžené kosodřeviny || snezka mountain in winter with the summit hut and snowy dwarf pines || mountain, snow, winter, building ||
realia | photo_landscape | Pravčická brána v Českém Švýcarsku při západu slunce || pravcicka gate, a natural sandstone arch in bohemian switzerland at sunset || rock, arch, forest, sunset ||
realia | photo_landscape | Český Krumlov s věží zámku nad meandrem Vltavy || cesky krumlov with the castle tower above the bend of the vltava river || castle, tower, river, town ||
realia | product_food | Svíčková na smetaně s knedlíky, brusinkami a šlehačkou na talíři || czech beef sirloin in cream sauce with bread dumplings, cranberries and whipped cream on a plate || food, plate, sauce, whipped cream ||
realia | product_food | Trdelník posypaný cukrem a skořicí na pražském trhu || trdelnik chimney cake sprinkled with sugar and cinnamon at a prague market || food, pastry, market ||
realia | product_food | Vánoční cukroví na talíři: vanilkové rohlíčky a linecké || czech christmas cookies on a plate: vanilla crescents and jam-filled linzer cookies || cookie, plate, christmas ||
realia | product_food | Vánoční kapr plave ve vaně || a christmas carp swimming in a bathtub || fish, bathtub, water ||
realia | fantasy_scifi | Rumcajs, loupežník z Řáholce, sedí pod dubem v lese || rumcajs, the kind fairytale robber with a beard and a feathered hat, sitting under an oak in a forest || beard, hat, forest, tree ||
realia | animals | Krteček z večerníčku vykukuje z krtince na louce || a little cartoon mole peeking out of a molehill on a meadow || mole (animal), grass, cartoon ||
realia | photo_landscape | Tramvaj T3 projíždí ulicí v Praze za deštivého večera || an old red and cream tatra t3 tram on a prague street on a rainy evening || train, street, rain, evening ||
realia | photo_landscape | Hrad Karlštejn nad lesy za podzimního rána || karlstejn gothic castle above autumn forests in the morning || castle, forest, autumn ||
realia | painting | Velikonoční kraslice v proutěném košíku, lidový ornament || czech easter eggs with folk ornaments in a wicker basket || easter egg, basket, pattern ||
realia | photo_portrait | Houbař s proutěným košíkem plným hřibů v jehličnatém lese || a mushroom picker with a wicker basket full of porcini in a conifer forest || basket, mushroom, forest ||
negation | photo_portrait | Žena bez brýlí čte knihu u okna || a woman reading a book by the window, not wearing glasses || reading, book, window || glasses
negation | photo_landscape | Lesní cesta bez lidí v mlze || an empty forest path in fog, no people || forest, path, fog, no humans || 1girl, 1boy, multiple girls, multiple boys
negation | animals | Pes bez obojku běží po pláži || a dog without a collar running on a beach || dog, running, beach || collar
negation | photo_portrait | Muž bez vousů v obleku, studiový portrét || clean-shaven man in a suit, studio portrait || suit, portrait || beard, mustache
negation | photo_landscape | Ulice starého města úplně bez aut || an old town street completely without cars || street, building || car, motor vehicle
negation | photo_landscape | Letní obloha bez jediného mráčku nad polem || a summer sky without a single cloud over a field || blue sky, field || cloud
negation | anime_character | Anime rytířka bez helmy, vlasy vlají ve větru || anime knight woman without a helmet, hair blowing in the wind || knight, armor, wind || helmet
negation | product_food | Talíř špaget bez příborů na stole || a plate of spaghetti on a table with no cutlery || spaghetti, plate, table || fork, spoon, knife
negation | photo_landscape | Moře při západu slunce, žádné lodě || the sea at sunset with no boats || ocean, sunset || boat, watercraft, ship
negation | anime_character | Anime žena s krátkými vlasy, ne dlouhými, v červené bundě || anime woman with short hair, not long hair, in a red jacket || short hair, red jacket || long hair
negation | photo_portrait | Portrét starého muže bez klobouku na venkově || portrait of an old man in the countryside without a hat || old man, portrait, outdoors || hat
negation | render_3d | 3D kuchyně, ve které nejsou žádné rostliny || 3d kitchen with no plants at all || 3d, kitchen || plant, potted plant
diacritics | photo_landscape | zena s cervenym destnikem na mostu v desti || woman with a red umbrella on a bridge in the rain || red umbrella, bridge, rain ||
diacritics | animals | cerna kocka sedi na zidi pri uplnku || black cat sitting on a wall under a full moon || black cat, wall, full moon ||
diacritics | product_food | fotka kafe a croissantu na stolku v kavarne, rano || photo of coffee and a croissant on a cafe table in the morning || coffee, croissant, table ||
diacritics | animals | hafan s klackem v hube na louce || a dog with a stick in its mouth on a meadow || dog, stick, grass ||
diacritics | photo_landscape | chalupa na horach v zime, z komina se kouri || a cottage in the mountains in winter with smoke from the chimney || cabin, snow, chimney, smoke ||
diacritics | fantasy_scifi | drak co chrli ohen na hrad, epicky || a dragon breathing fire onto a castle, epic || dragon, fire, castle ||
diacritics | photo_portrait | Babička peče buchty v kuchyni, all over mouka, útulno || a grandmother baking sweet buns in a cozy kitchen, flour everywhere || old woman, baking, kitchen, flour ||
diacritics | graphic | pixelart lodicka na mori pri zapadu slunce || pixel art of a small boat at sea at sunset || pixel art, boat, ocean, sunset ||
"""

# 15 few-shot inputs (NOT in EVAL) with one hand-written output per dialect:
#   style | czech || booru tags (Pony/Illustrious body, no quality/rating tags) || SDXL sentence (≤ 75 CLIP tokens) || FLUX prose
FEWSHOT = """
photo_portrait | Starý hodinář se zvětšovací lupou v oku opravuje kapesní hodinky || 1boy, solo, old man, watchmaker, monocle, magnifying glass, holding, pocket watch, workshop, desk, tools, indoors, warm lighting, realistic, photorealistic || close-up photo of an old watchmaker with a loupe in his eye repairing a pocket watch at a cluttered workbench, (warm lamp light:1.2), shallow depth of field, 85mm || A close-up photograph of an elderly watchmaker repairing a pocket watch at his workbench. A jeweler's loupe is pressed into one eye and tiny brass gears lie on a leather mat. Warm light from a desk lamp falls across his hands, while the shelves of clocks behind him dissolve into soft blur.
photo_landscape | Rybník v jižních Čechách za mlhavého podzimního rána, na hrázi staré duby || no humans, scenery, pond, lake, fog, morning, autumn, oak tree, tree, embankment, reflection, water, realistic, photorealistic || misty fishpond in south bohemia on an autumn morning, old oak trees on the dam, calm water reflections, soft golden light, landscape photo || A calm fishpond in South Bohemia on a misty autumn morning. Old oak trees with orange leaves line the earthen dam and their shapes reflect in the still grey water. Low fog drifts over the surface and the first golden sunlight breaks through it on the far shore.
product_food | Talíř bramboráků s česnekem a kyselou smetanou || no humans, food, potato pancake, plate, sour cream, garlic, table, still life, realistic, photorealistic || plate of crispy czech potato pancakes with garlic and a dollop of sour cream, rustic wooden table, (food photography:1.1), natural daylight || A rustic food photograph of a plate of golden, crispy Czech potato pancakes. A spoonful of sour cream and chopped garlic sit on top, and the plate rests on a worn wooden table. Soft daylight from the side shows the crunchy edges and a little steam rising.
animals | Bobr staví hráz na lesní řece || no humans, animal, beaver, building, dam, stick, river, forest, water, outdoors, realistic, photorealistic || a beaver building a dam of sticks on a forest river, wet fur, morning light, wildlife photo, telephoto || A wildlife photograph of a beaver dragging a branch onto its dam on a small forest river. Its wet brown fur glistens and the water ripples around the woven sticks. Morning light filters through the alders on the bank.
anime_character | Anime pošťačka na kole rozváží dopisy po kopcovitém městečku || 1girl, solo, mail carrier, bicycle, riding bicycle, bag, letter, hat, town, hill, outdoors, blue sky, smile || anime style mail carrier woman riding a bicycle with a bag of letters through a hilly little town, blue sky, cheerful, cel shading || An anime-style illustration of a cheerful young woman working as a mail carrier, riding a bicycle down a hilly little town street. A satchel full of letters hangs at her side and her cap is tilted back. Bright blue sky, crisp cel shading and clean line art.
anime_scenery | Anime nádraží v horách v zimě, svítí lampy a padá sníh || no humans, scenery, train station, mountain, winter, snow, snowing, lamp, light, night, platform, outdoors || anime background of a small mountain train station in winter at night, glowing lamps, falling snow, quiet platform, soft blue tones || An anime background painting of a small mountain railway station on a winter night. Warm lamps glow over an empty platform while snow falls steadily and settles on the roofs. Dark pine-covered slopes rise behind the station under a deep blue sky.
painting | Olejomalba žní na moravském poli, ženci a snopy || scenery, oil painting (medium), traditional media, field, wheat field, harvest, hay bale, people, summer, sky, painterly || oil painting of a harvest on a moravian wheat field, reapers and sheaves, golden summer light, (impasto:1.1), realist style || An oil painting of a summer harvest on a wide Moravian wheat field. Reapers work in the distance among golden sheaves tied with straw, under a high sky with towering clouds. Thick, textured brushstrokes in warm ochre and gold give the scene a realist nineteenth-century feel.
render_3d | 3D roztomilý ježek s jablkem na zádech, plastelína || no humans, 3d, clay, hedgehog, animal, apple, cute, simple background, studio lighting || cute 3d clay hedgehog carrying a red apple on its back, soft studio lighting, pastel background, (claymation style:1.2) || A cute claymation-style 3D render of a small hedgehog carrying a shiny red apple on its spiky back. The figure looks hand-sculpted from modelling clay, with visible fingerprints and soft rounded shapes. It stands on a pastel background under soft studio lighting.
graphic | Pixel art rybníka s kachnami a rákosím, letní den || no humans, pixel art, pond, duck, bird, reeds, water, summer, day, sky || pixel art of a pond with ducks and reeds on a sunny summer day, 16-bit retro game style, limited palette || A 16-bit pixel art scene of a small pond on a sunny summer day. Three ducks paddle among tall reeds and lily pads while white clouds drift across a bright blue sky. The image uses a limited retro-game palette with crisp, visible pixels.
fantasy_scifi | Hrad z ledu na vrcholu hory pod polární září || no humans, scenery, castle, ice, crystal, mountain, aurora, night sky, snow, fantasy || fantasy ice castle on a mountain peak under the northern lights, glowing crystal towers, snow, epic concept art || A fantasy concept art painting of a castle carved from blue ice on top of a snowy mountain peak. Its translucent towers glow from within while green and violet northern lights ripple across the night sky above. Wind lifts fine snow from the ridges below.
negation | Kočka bez ocasu sedí na židli || no humans, animal, cat, sitting, chair, indoors || a tailless cat sitting on a wooden chair, indoor photo, soft light || A photograph of a tailless cat sitting upright on a wooden kitchen chair, looking at the camera. The cat has no tail at all, just a short rounded rump. Soft indoor light comes from a nearby window.
negation | Pokoj bez nábytku, jen holé stěny a okno || no humans, scenery, empty room, room, indoors, window, wall, wooden floor, sunlight || an empty room with bare white walls, wooden floor and one window, sunlight, (no furniture:1.2), minimalist interior photo || A photograph of a completely empty room with bare white walls and a worn wooden floor. There is no furniture at all; the only feature is a single tall window letting a beam of afternoon sunlight fall across the floorboards.
idiom | Zahradník, který má zelené palce, mezi rajčaty || 1boy, solo, gardener, garden, tomato, plant, smile, outdoors, sunlight, realistic, photorealistic || a skilled gardener smiling among lush tomato plants in a thriving vegetable garden, sunny afternoon, photo || A photograph of a proud, skilled gardener smiling among tall, lush tomato plants heavy with ripe red fruit in a thriving vegetable garden. Afternoon sun warms the leaves. Nothing about his hands is unusual; the scene simply shows how well his garden grows.
realia | Pražský hrad nad Vltavou při západu slunce || no humans, scenery, castle, cathedral, river, bridge, city, sunset, sky, building || prague castle and st vitus cathedral above the vltava river at sunset, charles bridge in the foreground, warm golden light, cityscape photo || A cityscape photograph of Prague Castle and the Gothic spires of St. Vitus Cathedral rising above the Vltava River at sunset. The historic stone bridge crosses the river in the foreground and the old roofs glow orange in the warm evening light.
diacritics | zluta tramvaj v noci na nabrezi, odlesky || no humans, scenery, tram, train, night, river, street, light, reflection, city || yellow tram on a riverside street at night, wet cobblestones, reflections of lights, cinematic photo || A cinematic night photograph of a yellow tram gliding along a riverside street. Its lit windows and headlights reflect on the wet cobblestones and in the dark water of the river beside it.
"""

PONY_SOURCE = {"anime_character": "source_anime", "anime_scenery": "source_anime", "graphic": "source_cartoon",
               "render_3d": "source_cartoon"}


def eval_records():
    out = []
    for i, ln in enumerate([x.strip() for x in EVAL.strip().splitlines() if x.strip()], 1):
        head, ref_en, must, must_not = (s.strip() for s in ln.split("||"))
        cat, style, cz = (s.strip() for s in head.split("|", 2))
        out.append({"cid": f"cz{i:03d}", "category": cat, "style": style, "cz": cz, "ref_en": ref_en,
                    "must_tags": [t.strip() for t in must.split(",") if t.strip()],
                    "must_not": [t.strip() for t in must_not.split(",") if t.strip()],
                    "source": SOURCE, "license": LICENSE})
    return out


def fewshot_records():
    out = []
    for i, ln in enumerate([x.strip() for x in FEWSHOT.strip().splitlines() if x.strip()], 1):
        head, tags, sdxl, flux = (s.strip() for s in ln.split("||"))
        style, cz = (s.strip() for s in head.split("|", 1))
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        pony = ([PONY_SOURCE[style]] if style in PONY_SOURCE else []) + tag_list
        out.append({"fid": f"fs{i:02d}", "style": style, "cz": cz, "pony": pony, "illustrious": tag_list,
                    "sdxl": sdxl, "flux": flux, "source": SOURCE, "license": LICENSE})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", required=True)
    a = ap.parse_args()
    ev, fs = eval_records(), fewshot_records()
    with open(a.out, "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.fewshot, "w") as f:
        for r in fs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(ev)} eval prompts → {a.out}, {len(fs)} few-shot → {a.fewshot}", file=sys.stderr)
