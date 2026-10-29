"""UC18 prompt set: 300 SFW prompts, 10 styles × 30, each in two hand-written dialects.

Source: written for this benchmark by the agent (Claude, 2026-10-29), released CC0 — no gallery data,
no scraped prompts. Every prompt is SFW by construction (no nudity, violence, real people, brands or
minors; human figures are adults) and passes sfw.check_text() (test_uc18.py).

Line format:  prose || tags     (prose = natural English for SDXL/Juggernaut/FLUX,
                                 tags  = Danbooru-style for Pony/Illustrious/NoobAI/Animagine,
                                 without quality/rating tags — models.py adds those per checkpoint)
`python3 prompts_src.py --out prompts/uc18_prompts.jsonl` writes the JSONL used by gen_matrix.py.
"""
import argparse, json, sys

SOURCE = "written by Claude (agent) for UC18, 2026-10-29"
LICENSE = "CC0-1.0"

STYLES = {}

STYLES["photo_portrait"] = """
studio portrait of a middle-aged woman with short grey hair and round glasses, soft window light, neutral grey backdrop, 85mm photo || 1girl, solo, mature female, short hair, grey hair, round eyewear, glasses, portrait, upper body, simple background, grey background, realistic, photorealistic
close-up photo of an elderly fisherman with a white beard and a knitted cap, harbor in the background, overcast light || 1boy, solo, old man, beard, white hair, beanie, close-up, portrait, outdoors, harbor, boat, overcast, realistic, photorealistic
a young man with curly black hair laughing on a city rooftop at golden hour, denim jacket, shallow depth of field || 1boy, solo, curly hair, black hair, laughing, open mouth, denim jacket, rooftop, city, sunset, golden hour, depth of field, realistic, photorealistic
portrait of a woman with long red hair in a green wool coat standing in an autumn park, falling leaves, 50mm photo || 1girl, solo, long hair, red hair, green coat, standing, outdoors, park, autumn leaves, falling leaves, upper body, realistic, photorealistic
black and white photo of a jazz saxophonist playing on a dim club stage, smoke in the spotlight || 1boy, solo, saxophone, playing instrument, stage, spotlight, smoke, monochrome, greyscale, realistic, photorealistic
a chef in a white uniform plating a dessert in a busy restaurant kitchen, warm light, candid photo || 1boy, solo, chef, chef uniform, white shirt, holding, plate, food, kitchen, indoors, warm lighting, realistic, photorealistic
environmental portrait of a woman potter shaping clay on a wheel in a sunlit workshop, apron, dusty shelves || 1girl, solo, apron, pottery, clay, sitting, workshop, indoors, sunlight, shelf, realistic, photorealistic
a smiling elderly couple sitting on a wooden bench in a garden, holding hands, soft afternoon light || 1girl, 1boy, old woman, old man, smile, sitting, bench, holding hands, garden, flowers, outdoors, realistic, photorealistic
portrait of a man with a neatly trimmed beard wearing a tweed suit and flat cap, brick wall background, natural light || 1boy, solo, beard, suit, flat cap, formal, upper body, brick wall, realistic, photorealistic
a female astronaut in a white spacesuit holding her helmet, standing in a hangar, dramatic rim light || 1girl, solo, astronaut, spacesuit, holding helmet, standing, hangar, indoors, rim lighting, realistic, photorealistic
close-up portrait of a woman with freckles and braided hair looking at the camera, soft daylight, film grain || 1girl, solo, freckles, braid, brown hair, looking at viewer, close-up, portrait, film grain, realistic, photorealistic
a cyclist in a yellow rain jacket riding through a wet city street at night, neon reflections, motion blur || 1boy, solo, bicycle, riding, raincoat, yellow jacket, city, street, night, rain, neon lights, reflection, motion blur, realistic, photorealistic
a violinist woman in a long dark dress performing on a concert stage, warm spotlight, audience in shadow || 1girl, solo, violin, playing instrument, black dress, long dress, stage, spotlight, concert, realistic, photorealistic
portrait of a bearded mountaineer in a red jacket on a snowy ridge, sunglasses on his forehead, blue sky || 1boy, solo, beard, red jacket, eyewear on head, sunglasses, snow, mountain, blue sky, outdoors, upper body, realistic, photorealistic
a barista with tattooed forearms pouring latte art in a cozy cafe, steam rising, shallow focus || 1boy, solo, barista, tattoo, apron, holding cup, coffee, latte art, cafe, indoors, steam, depth of field, realistic, photorealistic
a woman reading a book in an armchair by a rainy window, knitted sweater, warm lamp light || 1girl, solo, reading, book, armchair, sitting, sweater, window, rain, lamp, indoors, realistic, photorealistic
headshot of a businesswoman with a short black bob in a navy blazer, clean white background, corporate photo || 1girl, solo, black hair, bob cut, blazer, formal, portrait, white background, simple background, realistic, photorealistic
an old blacksmith hammering glowing metal on an anvil, sparks flying, dark forge || 1boy, solo, old man, blacksmith, hammer, anvil, sparks, fire, indoors, dark, realistic, photorealistic
a gardener woman in a straw hat holding a basket of tomatoes in a greenhouse, morning sun || 1girl, solo, straw hat, holding basket, tomato, greenhouse, plant, sunlight, smile, realistic, photorealistic
portrait of a man in a white shirt standing in a wheat field at sunset, wind in his hair, backlit || 1boy, solo, white shirt, standing, wheat field, sunset, backlighting, wind, outdoors, realistic, photorealistic
a ballet dancer woman stretching at the barre in an empty rehearsal studio, morning light, side view || 1girl, solo, ballerina, leotard, stretching, ballet barre, dance studio, indoors, sunlight, from side, realistic, photorealistic
a street musician playing an accordion on a cobblestone square, pigeons around, soft evening light || 1boy, solo, accordion, playing instrument, cobblestone, town square, bird, pigeon, evening, outdoors, realistic, photorealistic
a scientist woman in a lab coat looking through a microscope, blue laboratory light, glassware || 1girl, solo, lab coat, microscope, laboratory, indoors, test tube, blue theme, realistic, photorealistic
portrait of a smiling man with a grey beard wearing a turban, warm studio light, dark background || 1boy, solo, old man, grey hair, beard, turban, smile, portrait, dark background, simple background, realistic, photorealistic
a woman hiking with a backpack on a mountain trail, pine forest, mist in the valley, wide shot || 1girl, solo, hiking, backpack, walking, mountain, pine tree, forest, fog, wide shot, outdoors, realistic, photorealistic
a firefighter in full gear resting against a red truck, helmet under his arm, tired smile || 1boy, solo, firefighter, uniform, helmet, holding helmet, fire truck, leaning, smile, outdoors, realistic, photorealistic
portrait of a woman with silver hair in a bun, pearl earrings, dark green velvet dress, rembrandt lighting || 1girl, solo, old woman, grey hair, hair bun, earrings, pearl earrings, green dress, velvet, portrait, dark background, realistic, photorealistic
a fisherwoman in yellow oilskins hauling a net on a small boat, choppy grey sea || 1girl, solo, raincoat, yellow coat, fishing net, boat, ocean, waves, overcast, outdoors, realistic, photorealistic
a man in a wool sweater drinking tea at a wooden table in a mountain cabin, steam, window view of snow || 1boy, solo, sweater, drinking, cup, tea, table, cabin, indoors, window, snow, realistic, photorealistic
portrait of a woman in a yellow raincoat holding a transparent umbrella on a rainy street, bokeh lights || 1girl, solo, raincoat, yellow coat, holding umbrella, transparent umbrella, rain, street, bokeh, night, realistic, photorealistic
"""

STYLES["photo_landscape"] = """
a misty pine forest at dawn with sun rays breaking through the trees, landscape photo || no humans, scenery, forest, pine tree, fog, sunrise, sunlight, light rays, nature, realistic, photorealistic
a turquoise mountain lake surrounded by snowy peaks, reflection on still water, wide angle || no humans, scenery, lake, mountain, snow, reflection, water, blue sky, wide shot, nature, realistic, photorealistic
an old stone bridge over a calm river in a medieval town at blue hour, warm street lamps || no humans, scenery, bridge, stone bridge, river, town, medieval, evening, street lamp, building, realistic, photorealistic
desert sand dunes with long shadows at sunset, rippled sand, clear sky || no humans, scenery, desert, sand, dune, sunset, shadow, sky, realistic, photorealistic
a lighthouse on a rocky cliff during a storm, crashing waves, dark clouds || no humans, scenery, lighthouse, cliff, rock, ocean, waves, storm, dark clouds, realistic, photorealistic
a narrow cobblestone alley in an old european town with flower boxes on the windows, morning light || no humans, scenery, alley, cobblestone, building, window, flower, potted plant, town, morning, realistic, photorealistic
aerial photo of terraced rice fields in green and gold, morning mist || no humans, scenery, aerial view, field, rice paddy, terraced field, green theme, fog, morning, realistic, photorealistic
a snowy village with wooden houses and smoking chimneys at dusk, lights in the windows || no humans, scenery, village, snow, house, chimney, smoke, dusk, window, light, winter, realistic, photorealistic
a waterfall in a lush green jungle with ferns and mossy rocks, long exposure || no humans, scenery, waterfall, jungle, fern, moss, rock, plant, water, nature, realistic, photorealistic
the northern lights over a frozen lake with a small cabin, starry sky || no humans, scenery, aurora, night sky, starry sky, lake, frozen lake, ice, cabin, snow, realistic, photorealistic
a modern glass skyscraper reflecting clouds, low angle view, minimal composition || no humans, scenery, skyscraper, building, glass, reflection, cloud, sky, from below, city, realistic, photorealistic
a lavender field in rows leading to a stone farmhouse, summer evening light || no humans, scenery, field, lavender, flower field, farmhouse, house, summer, evening, sky, realistic, photorealistic
a rainy tokyo-style street at night with glowing shop signs and wet asphalt reflections || no humans, scenery, city, street, night, rain, neon lights, sign, reflection, road, realistic, photorealistic
a sandy tropical beach with palm trees and turquoise water, white clouds, midday || no humans, scenery, beach, palm tree, ocean, sand, cloud, blue sky, day, tropical, realistic, photorealistic
autumn forest road covered with orange leaves, soft fog, empty road || no humans, scenery, road, forest, autumn, autumn leaves, orange theme, fog, tree, realistic, photorealistic
a gothic cathedral interior with stained glass windows and light beams on stone columns || no humans, scenery, church, cathedral, indoors, stained glass, light rays, pillar, architecture, realistic, photorealistic
a small red wooden cabin on a green hill under a dramatic cloudy sky, rolling hills || no humans, scenery, cabin, house, hill, grass, cloud, cloudy sky, nature, realistic, photorealistic
a canal in an old city with boats and colorful houses, reflections at sunset || no humans, scenery, canal, boat, house, building, water, reflection, sunset, city, realistic, photorealistic
a frozen waterfall with blue icicles in a winter canyon, overcast light || no humans, scenery, waterfall, frozen, ice, icicle, canyon, winter, snow, realistic, photorealistic
a vineyard on rolling hills at sunrise, rows of grapevines, distant village || no humans, scenery, vineyard, grapes, hill, sunrise, village, field, sky, realistic, photorealistic
a mountain road with hairpin turns seen from above, green alpine meadows || no humans, scenery, road, mountain, aerial view, from above, grass, alpine, nature, realistic, photorealistic
an abandoned greenhouse overgrown with plants, broken glass panes, soft green light || no humans, scenery, greenhouse, plant, overgrown, ruins, broken glass, indoors, green theme, realistic, photorealistic
a city skyline at night across a river, long exposure light trails on a bridge || no humans, scenery, cityscape, skyline, night, river, bridge, light trail, building, realistic, photorealistic
a field of sunflowers under a bright blue summer sky, one tall sunflower in front || no humans, scenery, sunflower, flower field, blue sky, summer, cloud, realistic, photorealistic
a quiet japanese garden with a red wooden bridge, koi pond and maple trees || no humans, scenery, japanese garden, bridge, red bridge, pond, koi, maple tree, nature, realistic, photorealistic
a basalt sea stack on a black sand beach, foggy morning, long exposure sea || no humans, scenery, beach, black sand, rock, ocean, fog, morning, realistic, photorealistic
a cozy bookshop street front at night with warm windows and a bicycle parked outside || no humans, scenery, shop, bookshop, street, night, window, bicycle, light, building, realistic, photorealistic
a canyon with red rock walls and a winding river, midday sun || no humans, scenery, canyon, rock, cliff, river, desert, day, sunlight, realistic, photorealistic
a misty bamboo forest path with soft green light || no humans, scenery, bamboo, bamboo forest, path, fog, green theme, nature, realistic, photorealistic
a harbor with fishing boats at sunrise, calm water, seagulls in the sky || no humans, scenery, harbor, boat, ocean, sunrise, bird, seagull, water, realistic, photorealistic
"""

STYLES["product_food"] = """
a steaming bowl of ramen with a soft boiled egg, green onions and nori on a dark wooden table, food photography || no humans, food, ramen, bowl, egg, soft-boiled egg, nori, green onion, chopsticks, table, steam, still life, realistic, photorealistic
a stack of fluffy pancakes with blueberries and maple syrup dripping, morning light, overhead shot || no humans, food, pancake, stack, blueberry, syrup, plate, from above, still life, realistic, photorealistic
a glass of iced coffee with condensation on a marble counter, minimal product shot || no humans, cup, drinking glass, iced coffee, ice, ice cube, marble, still life, simple background, realistic, photorealistic
a vintage film camera on a wooden desk next to a notebook and a pen, soft window light || no humans, camera, still life, desk, notebook, pen, wooden table, window, sunlight, realistic, photorealistic
a slice of strawberry cake with whipped cream on a white plate, pastel background || no humans, food, cake, cake slice, strawberry, whipped cream, plate, pastel colors, simple background, realistic, photorealistic
a rustic loaf of sourdough bread on a linen cloth with a bread knife, flour dust || no humans, food, bread, knife, cloth, flour, table, still life, realistic, photorealistic
a pair of white sneakers floating on a pastel blue background, studio product photo || no humans, shoes, sneakers, white footwear, floating, blue background, simple background, still life, realistic, photorealistic
a luxury wristwatch with a leather strap on black slate, dramatic side light, macro || no humans, watch, wristwatch, still life, black background, close-up, realistic, photorealistic
a colorful bowl of fresh fruit salad with kiwi, mango and pomegranate seeds, top view || no humans, food, fruit, bowl, kiwi (fruit), mango, pomegranate, from above, still life, realistic, photorealistic
a ceramic teapot and two cups on a bamboo tray, steam rising, zen mood || no humans, teapot, cup, tea, tray, steam, still life, table, realistic, photorealistic
a neapolitan pizza with basil and mozzarella fresh out of a wood oven, glowing embers || no humans, food, pizza, basil, cheese, oven, fire, still life, realistic, photorealistic
a glass perfume bottle on a mirror surface with soft pink light and reflections || no humans, perfume bottle, bottle, still life, reflection, pink theme, simple background, realistic, photorealistic
a bowl of tomato soup with croutons and a swirl of cream, wooden spoon, autumn table || no humans, food, soup, bowl, spoon, table, still life, realistic, photorealistic
a pair of headphones on a desk next to a laptop and a potted succulent, clean workspace || no humans, headphones, laptop, desk, potted plant, plant, still life, indoors, realistic, photorealistic
a plate of sushi rolls and nigiri on a slate board with chopsticks and soy sauce || no humans, food, sushi, chopsticks, plate, still life, table, realistic, photorealistic
a chocolate bar broken into pieces with cocoa beans and cocoa powder, dark mood || no humans, food, chocolate, chocolate bar, still life, dark background, realistic, photorealistic
an old leather backpack on a wooden bench in a forest, morning light, product shot || no humans, backpack, bag, bench, forest, sunlight, still life, outdoors, realistic, photorealistic
a mug of hot chocolate with marshmallows next to a knitted scarf and pine cones, winter mood || no humans, food, cup, mug, hot chocolate, marshmallow, scarf, pine cone, still life, realistic, photorealistic
a bouquet of tulips in a glass vase on a kitchen table, morning sunlight || no humans, flower, tulip, bouquet, vase, table, sunlight, still life, indoors, realistic, photorealistic
a burger with melted cheese, lettuce and tomato on a wooden board, fries on the side || no humans, food, hamburger, cheese, lettuce, tomato, french fries, still life, realistic, photorealistic
a glass jar of honey with a wooden dipper, honey dripping, warm backlight || no humans, food, honey, jar, still life, backlighting, table, realistic, photorealistic
a mechanical keyboard with colorful keycaps on a dark desk, rgb glow || no humans, keyboard (computer), desk, still life, dark, glowing, realistic, photorealistic
a basket of fresh vegetables: carrots, leeks, radishes and cabbage, farmers market || no humans, food, vegetable, carrot, basket, still life, outdoors, realistic, photorealistic
a pair of reading glasses resting on an open book with a cup of coffee, cozy light || no humans, glasses, book, open book, cup, coffee, still life, table, realistic, photorealistic
a macaron tower in pastel colors on a cake stand, bakery window || no humans, food, macaron, pastel colors, cake stand, still life, realistic, photorealistic
an espresso machine pouring a shot into a small cup, close-up, cafe background bokeh || no humans, coffee, cup, coffee maker, close-up, bokeh, still life, realistic, photorealistic
a bowl of colorful poke with salmon, avocado and edamame, top view on a light table || no humans, food, bowl, salmon, avocado, from above, still life, realistic, photorealistic
a candle burning in a glass holder next to dried flowers on a linen tablecloth || no humans, candle, fire, flower, dried flower, tablecloth, still life, realistic, photorealistic
a red bicycle leaning against a pastel yellow wall, sunny street, product style || no humans, bicycle, wall, yellow wall, street, outdoors, sunlight, realistic, photorealistic
a set of watercolor paints and brushes on a sheet of paper with a half-painted landscape || no humans, paintbrush, paint, paper, art supplies, still life, desk, realistic, photorealistic
"""

STYLES["animals"] = """
a red fox sitting in fresh snow, winter forest, soft light, wildlife photo || no humans, animal, fox, sitting, snow, winter, forest, outdoors, realistic, photorealistic
a golden retriever puppy running through a meadow with flowers, sunny day || no humans, animal, dog, golden retriever, puppy, running, field, flower, day, outdoors, realistic, photorealistic
a barn owl perched on a wooden fence post at dusk, wings slightly open || no humans, animal, bird, owl, perched, fence, dusk, outdoors, realistic, photorealistic
a ginger cat sleeping on a sunny windowsill next to a potted plant || no humans, animal, cat, orange cat, sleeping, windowsill, sunlight, potted plant, indoors, realistic, photorealistic
a herd of elephants walking across the savanna at sunset, dust in the air || no humans, animal, elephant, herd, savannah, sunset, walking, dust, outdoors, realistic, photorealistic
a hummingbird hovering next to a red flower, frozen wings, macro || no humans, animal, bird, hummingbird, flying, flower, red flower, close-up, realistic, photorealistic
a sea turtle swimming over a coral reef, sun rays in clear blue water || no humans, animal, turtle, sea turtle, swimming, underwater, coral, coral reef, light rays, realistic, photorealistic
a fluffy white rabbit sitting in green grass with dandelions || no humans, animal, rabbit, white fur, sitting, grass, dandelion, outdoors, realistic, photorealistic
a majestic lion resting on a rock, golden mane, warm evening light || no humans, animal, lion, lying, rock, sunset, outdoors, realistic, photorealistic
a pair of penguins standing on ice with the ocean behind them, overcast || no humans, animal, penguin, standing, ice, ocean, overcast, outdoors, realistic, photorealistic
a squirrel holding a hazelnut on a mossy branch, autumn forest background || no humans, animal, squirrel, holding, nut, branch, moss, autumn, forest, realistic, photorealistic
a horse galloping on a beach at sunrise, splashing water || no humans, animal, horse, running, galloping, beach, sunrise, water, splashing, realistic, photorealistic
a chameleon on a branch with vivid green and orange skin, macro photo || no humans, animal, chameleon, lizard, branch, close-up, green theme, realistic, photorealistic
a family of ducks swimming in a calm pond, reflections, morning mist || no humans, animal, bird, duck, swimming, pond, water, reflection, fog, morning, realistic, photorealistic
a snow leopard on a rocky mountain slope, piercing eyes, snowfall || no humans, animal, snow leopard, rock, mountain, snowing, outdoors, realistic, photorealistic
a corgi wearing a tiny raincoat sitting on a wet sidewalk, cute || no humans, animal, dog, corgi, raincoat, sitting, rain, sidewalk, outdoors, realistic, photorealistic
a monarch butterfly on a purple flower, shallow depth of field || no humans, animal, butterfly, insect, flower, purple flower, depth of field, close-up, realistic, photorealistic
a humpback whale breaching out of the ocean, spray, dramatic sky || no humans, animal, whale, jumping, ocean, splashing, sky, cloud, realistic, photorealistic
a hedgehog walking on fallen leaves in an autumn garden || no humans, animal, hedgehog, walking, autumn leaves, garden, outdoors, realistic, photorealistic
a flamingo standing on one leg in a shallow lagoon, pink reflection || no humans, animal, bird, flamingo, standing on one leg, water, lagoon, reflection, pink theme, realistic, photorealistic
a black cat with yellow eyes sitting in a doorway, moody light || no humans, animal, cat, black cat, yellow eyes, sitting, doorway, dark, realistic, photorealistic
a red panda climbing a tree branch, fluffy tail, forest background || no humans, animal, red panda, climbing, tree, branch, forest, outdoors, realistic, photorealistic
a pod of dolphins jumping out of blue water near the coast || no humans, animal, dolphin, jumping, ocean, water, splashing, day, realistic, photorealistic
a highland cow with long shaggy fur and horns standing in a misty field || no humans, animal, cow, horns, fur, standing, field, fog, outdoors, realistic, photorealistic
a parrot with bright blue and yellow feathers on a branch in the rainforest || no humans, animal, bird, parrot, macaw, perched, branch, jungle, realistic, photorealistic
a curious meerkat standing upright on a sand mound, desert background || no humans, animal, meerkat, standing, sand, desert, outdoors, realistic, photorealistic
a jellyfish glowing in dark water, translucent tentacles || no humans, animal, jellyfish, underwater, glowing, dark background, realistic, photorealistic
a wolf howling on a hill under a full moon, night || no humans, animal, wolf, howling, hill, full moon, night, night sky, outdoors, realistic, photorealistic
a sleepy koala hugging a eucalyptus branch || no humans, animal, koala, sleeping, hugging, branch, tree, outdoors, realistic, photorealistic
a goldfish in a round glass bowl on a sunny table, water caustics || no humans, animal, fish, goldfish, fishbowl, water, table, sunlight, realistic, photorealistic
"""

STYLES["anime_character"] = """
anime illustration of a young woman with long silver hair and blue eyes in a white coat, standing in falling snow || 1girl, solo, long hair, silver hair, blue eyes, white coat, standing, snowing, outdoors, looking at viewer, anime coloring
anime style knight woman with short blonde hair in silver armor holding a sword, castle courtyard || 1girl, solo, knight, short hair, blonde hair, armor, silver armor, holding sword, sword, castle, outdoors, cowboy shot
anime illustration of a cheerful waitress with a ponytail carrying a tray of coffee in a cafe || 1girl, solo, ponytail, brown hair, waitress, apron, holding tray, coffee, cafe, indoors, smile
anime man with messy black hair and a red scarf standing on a train platform at sunset || 1boy, solo, messy hair, black hair, red scarf, standing, train station, platform, sunset, outdoors
anime style mage woman with purple hair casting a glowing blue spell, flowing robes, dark library || 1girl, solo, mage, purple hair, long hair, robe, magic, casting spell, glowing, library, bookshelf, indoors
anime illustration of a woman in a yellow raincoat with an umbrella jumping over a puddle || 1girl, solo, raincoat, yellow raincoat, holding umbrella, jumping, puddle, rain, outdoors, smile
anime man in a dark suit and tie with glasses reading documents in an office at night || 1boy, solo, suit, black suit, necktie, glasses, holding paper, office, indoors, night, desk
anime style archer woman with green hair aiming a bow in a forest, leaves flying || 1girl, solo, archer, green hair, holding bow (weapon), bow (weapon), arrow (projectile), aiming, forest, outdoors, leaf
anime illustration of a pilot woman in a leather jacket and goggles next to a small propeller plane || 1girl, solo, pilot, leather jacket, goggles on head, standing, airplane, propeller, outdoors, sky
anime style samurai man with a long ponytail in a dark kimono, cherry blossoms falling || 1boy, solo, samurai, long hair, ponytail, black hair, kimono, katana, cherry blossoms, petals, outdoors
anime woman with pink twin tails in a casual hoodie playing a video game on a couch, cozy room || 1girl, solo, twintails, pink hair, hoodie, holding controller, game controller, playing games, couch, sitting, indoors
anime illustration of a detective in a trench coat and hat holding a magnifying glass, foggy street || 1boy, solo, detective, trench coat, fedora, holding magnifying glass, magnifying glass, street, fog, night
anime style woman with short brown hair painting on a canvas in a sunny studio, paint on her cheek || 1girl, solo, short hair, brown hair, painting (action), canvas (object), paintbrush, holding paintbrush, paint on face, art studio, indoors, sunlight
anime man with white hair and golden eyes in a black coat on a rooftop under a full moon || 1boy, solo, white hair, yellow eyes, black coat, rooftop, full moon, night, night sky, city, standing
anime illustration of a florist woman arranging flowers in a small shop, apron, smile || 1girl, solo, florist, apron, holding flower, flower, bouquet, flower shop, indoors, smile
anime style space explorer woman in a white suit floating near a window with earth outside || 1girl, solo, astronaut, spacesuit, floating, space station, window, earth (planet), indoors
anime man with red spiky hair in a chef apron flipping a pan with flames, restaurant kitchen || 1boy, solo, red hair, spiky hair, chef, apron, holding pan, frying pan, fire, kitchen, indoors
anime illustration of a woman in a blue kimono holding a paper lantern at a summer festival at night || 1girl, solo, kimono, blue kimono, holding lantern, paper lantern, summer festival, night, outdoors, fireworks
anime style mechanic woman with goggles and grease on her face repairing a motorcycle in a garage || 1girl, solo, mechanic, goggles on head, overalls, wrench, holding wrench, motorcycle, garage, indoors, dirty face
anime man playing an acoustic guitar on a park bench, autumn trees || 1boy, solo, guitar, acoustic guitar, playing instrument, sitting, bench, park, autumn, outdoors
anime illustration of a librarian woman with glasses and a long braid climbing a ladder for a book || 1girl, solo, glasses, long hair, single braid, ladder, holding book, library, bookshelf, indoors, cardigan
anime style woman in a white lab coat with a holographic display, futuristic laboratory || 1girl, solo, lab coat, hologram, holographic interface, laboratory, indoors, science fiction
anime man with a bandana and a backpack hiking on a mountain path with a view of clouds || 1boy, solo, bandana, backpack, hiking, mountain, path, cloud, sky, outdoors, from behind
anime illustration of a witch woman flying on a broom over a town at dusk with a black cat || 1girl, solo, witch, witch hat, black dress, riding broom, broom, flying, town, dusk, black cat, cat
anime style barista man making latte art, warm cafe light, apron || 1boy, solo, barista, apron, holding cup, coffee, latte art, cafe, indoors, warm lighting
anime woman with long black hair and red eyes in a gothic dress holding a candle in a dark hallway || 1girl, solo, long hair, black hair, red eyes, gothic, black dress, holding candle, candle, hallway, dark, indoors
anime illustration of two friends, a woman and a man, sharing an umbrella in the rain at a bus stop || 1girl, 1boy, sharing umbrella, umbrella, rain, bus stop, outdoors, standing, smile
anime style ninja woman in dark clothes crouching on a rooftop at night, moonlight || 1girl, solo, ninja, black clothes, mask, crouching, rooftop, night, moonlight, full moon, outdoors
anime man in a lab coat with messy hair holding a glowing potion flask, alchemy workshop || 1boy, solo, lab coat, messy hair, holding, flask, potion, glowing, alchemy, workshop, indoors
anime illustration of a sailor woman with a captain hat at the helm of a ship, ocean waves || 1girl, solo, captain, hat, peaked cap, ship's wheel, ship, ocean, waves, outdoors, coat
"""

STYLES["anime_scenery"] = """
anime background of a quiet rural train station in summer, cumulonimbus clouds, cicadas mood || no humans, scenery, train station, rural, summer, cloud, cumulonimbus cloud, blue sky, power lines, outdoors
anime scenery of a cozy wooden ramen stall at night with lanterns and steam, empty street || no humans, scenery, food stand, ramen, lantern, night, street, steam, outdoors
anime style floating islands with waterfalls in a bright blue sky, fantasy landscape || no humans, scenery, floating island, waterfall, sky, cloud, blue sky, fantasy, outdoors
anime background of a classroom at sunset, empty desks, orange light through windows || no humans, scenery, classroom, desk, chair, window, sunset, orange sky, indoors
anime scenery of a shrine gate in a forest with cherry blossoms and stone lanterns || no humans, scenery, torii, shrine, forest, cherry blossoms, stone lantern, stairs, outdoors
anime style night city view from a rooftop with neon signs and rain || no humans, scenery, city, cityscape, rooftop, night, neon lights, rain, building, sign
anime background of a sunflower field with a winding dirt road and a white house || no humans, scenery, sunflower, field, road, house, summer, blue sky, cloud, outdoors
anime scenery of a cozy bedroom with plants, a desk lamp and a rainy window at night || no humans, scenery, bedroom, indoors, plant, desk lamp, window, rain, night, bed
anime style seaside town with steep stairs and orange roofs, sparkling ocean || no humans, scenery, town, stairs, roof, ocean, coastline, day, blue sky, outdoors
anime background of a snowy forest with a frozen river and a small bridge, twilight || no humans, scenery, forest, snow, winter, river, frozen, bridge, twilight, outdoors
anime scenery of a starry night sky over a hill with a lone tree, milky way || no humans, scenery, night sky, starry sky, milky way, hill, tree, grass, night, outdoors
anime style abandoned amusement park with a rusty ferris wheel overgrown with flowers || no humans, scenery, amusement park, ferris wheel, ruins, overgrown, flower, cloud, outdoors
anime background of a convenience store at night with glowing lights and a vending machine || no humans, scenery, convenience store, night, vending machine, light, street, outdoors
anime scenery of a train passing over the sea on a bridge, clear sky, summer || no humans, scenery, train, railroad tracks, bridge, ocean, summer, blue sky, cloud, outdoors
anime style magical library with endless bookshelves and floating books, warm light || no humans, scenery, library, bookshelf, book, floating object, indoors, fantasy, warm lighting
anime background of a mountain village with rice fields and mist at dawn || no humans, scenery, village, mountain, rice paddy, field, fog, dawn, outdoors, house
anime scenery of an autumn park with a pond, red maple leaves and a wooden bench || no humans, scenery, park, pond, autumn leaves, maple tree, bench, autumn, outdoors
anime style cyberpunk alley with holograms, cables and puddles reflecting pink light || no humans, scenery, alley, cyberpunk, hologram, cable, puddle, reflection, neon lights, night
anime background of a beach at sunset with a lifeguard tower and gentle waves || no humans, scenery, beach, sunset, ocean, waves, lifeguard tower, sand, outdoors
anime scenery of a greenhouse cafe full of hanging plants and sunlight || no humans, scenery, greenhouse, cafe, plant, hanging plant, sunlight, indoors, table, chair
anime style castle on a cliff above the clouds at sunrise || no humans, scenery, castle, cliff, above clouds, cloud, sunrise, sky, fantasy, outdoors
anime background of a rainy bus stop on a country road with a forest behind || no humans, scenery, bus stop, rain, road, forest, rural, outdoors, overcast
anime scenery of a lighthouse on an island under a sky full of shooting stars || no humans, scenery, lighthouse, island, ocean, night sky, shooting star, starry sky, night
anime style old bookshop interior with stacks of books and a cat sleeping on the counter || no humans, scenery, bookshop, indoors, book, book stack, cat, sleeping, counter
anime background of a riverside with cherry blossom trees and a small boat, spring || no humans, scenery, river, cherry blossoms, tree, boat, spring (season), petals, outdoors
anime scenery of a desert with ancient ruins and two moons in the sky || no humans, scenery, desert, ruins, sand, two moons, moon, sky, fantasy, outdoors
anime style kitchen in the morning with sunlight, breakfast on the table and a kettle || no humans, scenery, kitchen, indoors, morning, sunlight, table, food, kettle, window
anime background of an underwater city with glowing domes and fish || no humans, scenery, underwater, city, dome, fish, glowing, bubble, fantasy
anime scenery of a mountain pass with a stone path, prayer flags and snowy peaks || no humans, scenery, mountain, path, stone, flag, snow, sky, cloud, outdoors
anime style giant tree with a treehouse and lanterns at dusk || no humans, scenery, tree, treehouse, lantern, dusk, sky, fantasy, outdoors
"""

STYLES["painting"] = """
oil painting of a stormy sea with a sailing ship, dramatic waves, romantic era style || no humans, scenery, oil painting (medium), traditional media, ocean, ship, sailboat, waves, storm, cloud, painterly
watercolor painting of a quiet harbor village with pastel houses, loose brush strokes || no humans, scenery, watercolor (medium), traditional media, harbor, house, boat, water, pastel colors, painterly
impressionist painting of a woman with a parasol in a poppy field, sunny afternoon || 1girl, solo, impressionism, oil painting (medium), traditional media, holding parasol, parasol, flower field, poppy (flower), dress, outdoors, painterly
ukiyo-e style woodblock print of a great wave with mount fuji in the background || no humans, ukiyo-e, traditional media, waves, ocean, mount fuji, mountain, boat, sky
oil painting still life of apples, a copper jug and a loaf of bread on a dark table, dutch masters style || no humans, still life, oil painting (medium), traditional media, apple, jug, bread, table, dark background, painterly
watercolor illustration of a fox in a winter forest, soft washes, white paper || no humans, animal, fox, watercolor (medium), traditional media, forest, winter, snow, white background
gouache painting of a cozy cafe interior with warm colors and plants || no humans, scenery, gouache (medium), traditional media, cafe, indoors, plant, table, warm colors
oil portrait of an elderly man with a pipe in the style of rembrandt, chiaroscuro || 1boy, solo, old man, beard, smoking pipe, portrait, oil painting (medium), traditional media, dark background, chiaroscuro
chinese ink wash painting of misty mountains, a small boat and pine trees || no humans, scenery, ink wash painting, traditional media, mountain, fog, boat, pine tree, monochrome
pastel drawing of a ballet dancer woman tying her shoes, degas inspired || 1girl, solo, ballerina, pastel (medium), traditional media, sitting, tutu, ballet slippers, tying footwear, indoors
watercolor botanical illustration of a sprig of lavender and bees, white background || no humans, watercolor (medium), traditional media, botanical, lavender, flower, bee, insect, white background
oil painting of a sunflower field under a swirling starry sky, expressive brush strokes || no humans, scenery, oil painting (medium), traditional media, sunflower, field, starry sky, night, painterly, impasto
acrylic painting of a colorful hot air balloon festival over green hills || no humans, scenery, acrylic paint (medium), traditional media, hot air balloon, hill, sky, cloud, colorful
watercolor painting of a rainy paris street with an umbrella crowd, reflections || scenery, watercolor (medium), traditional media, city, street, rain, umbrella, reflection, building, painterly
art nouveau poster of a woman with flowing hair surrounded by flowers and ornaments || 1girl, solo, art nouveau, long hair, flower, ornament, poster (medium), portrait, flowing hair
oil painting of a medieval market square with merchants and stalls, warm afternoon light || scenery, oil painting (medium), traditional media, market, market stall, medieval, town, crowd, painterly
charcoal drawing of a lion head, expressive strokes, textured paper || no humans, animal, lion, charcoal (medium), traditional media, monochrome, greyscale, portrait, sketch
watercolor painting of a cherry blossom branch with small birds || no humans, watercolor (medium), traditional media, cherry blossoms, branch, bird, white background, painterly
oil painting of a lighthouse at sunset with orange clouds, palette knife texture || no humans, scenery, oil painting (medium), traditional media, lighthouse, sunset, cloud, ocean, impasto
pointillism painting of a park by a river with people strolling on a sunday || scenery, pointillism, traditional media, park, river, tree, people, outdoors, painterly
gouache illustration of a mountain cabin in autumn with smoke from the chimney || no humans, scenery, gouache (medium), traditional media, cabin, mountain, autumn, smoke, chimney, tree
oil painting of a vase of peonies on a windowsill, soft light, classical || no humans, still life, oil painting (medium), traditional media, flower, peony (flower), vase, windowsill, painterly
watercolor portrait of a woman with a sun hat, loose wet-on-wet style || 1girl, solo, watercolor (medium), traditional media, sun hat, portrait, smile, painterly
byzantine style mosaic of a peacock with gold background || no humans, animal, bird, peacock, mosaic, gold background, ornament, traditional media
oil painting of a snowy winter village with skaters on a frozen pond, bruegel style || scenery, oil painting (medium), traditional media, village, snow, winter, ice skating, frozen pond, people, painterly
watercolor sketch of a vintage car parked on a sunny street, ink lines || no humans, watercolor (medium), ink (medium), traditional media, car, vintage car, street, sketch
oil painting of a dramatic mountain landscape with a waterfall and golden light, hudson river school || no humans, scenery, oil painting (medium), traditional media, mountain, waterfall, sunlight, cloud, painterly
tempera painting of a medieval angel with gold halo, flat perspective || 1girl, solo, angel, halo, wings, robe, gold, tempera, traditional media, medieval, painterly
pastel painting of a field of lupins by the sea, soft colors || no humans, scenery, pastel (medium), traditional media, flower field, lupine, ocean, sky, pastel colors
oil painting of a quiet library with a reading lamp and old books, warm glow || no humans, scenery, oil painting (medium), traditional media, library, lamp, book, bookshelf, indoors, warm lighting
"""

STYLES["render_3d"] = """
3d render of a cute isometric tiny house on a floating island with trees, soft lighting, pastel colors || no humans, 3d, isometric, house, floating island, tree, pastel colors, blender (medium), simple background
3d clay render of a smiling cartoon cactus in a pot, soft studio light || no humans, 3d, clay, cactus, potted plant, smile, cartoon, simple background, studio lighting
low poly 3d render of a mountain landscape with a lake and pine trees at sunset || no humans, 3d, low poly, scenery, mountain, lake, pine tree, sunset
3d render of a futuristic sports car in a dark studio, glossy reflections, rim light || no humans, 3d, car, sports car, futuristic, dark background, reflection, rim lighting, realistic
3d render of a cozy isometric bedroom with a bed, desk, plants and fairy lights || no humans, 3d, isometric, bedroom, bed, desk, plant, string lights, indoors, cozy
3d character render of a friendly robot with big round eyes waving, white background || no humans, 3d, robot, waving, big eyes, cute, white background, simple background
3d render of a glass sphere filled with a tiny forest, studio lighting, caustics || no humans, 3d, glass, sphere, terrarium, forest, tree, studio lighting, simple background
3d render of a stylized fantasy mushroom village glowing at night || no humans, 3d, mushroom house, village, glowing, night, fantasy, stylized
3d render of a ceramic coffee mug with steam on a wooden table, photoreal product visualization || no humans, 3d, cup, mug, coffee, steam, wooden table, realistic, still life
3d render of a cartoon astronaut hamster in a spacesuit floating among planets || no humans, 3d, hamster, animal, spacesuit, astronaut, floating, space, planet, cartoon, cute
3d isometric render of a small city block with shops, trees and cars || no humans, 3d, isometric, city, building, shop, tree, car, road
3d render of a crystal geode cut in half with purple amethyst, macro, dark background || no humans, 3d, crystal, amethyst, gem, close-up, dark background, purple theme
3d render of a whimsical candy land with lollipop trees and chocolate river || no humans, 3d, candy, lollipop, chocolate, river, colorful, fantasy, cartoon
3d render of a vintage radio on a shelf, warm light, subsurface wood texture || no humans, 3d, radio, shelf, warm lighting, vintage, still life, realistic
3d clay animation style fox character holding a lantern in a snowy forest || no humans, 3d, clay, fox, animal, holding lantern, lantern, snow, forest, cute
3d render of a modern minimalist living room with large windows and sunlight, archviz || no humans, 3d, living room, indoors, window, sunlight, couch, minimalism, realistic
3d render of a cute dragon toy made of plastic, glossy, bright colors, white background || no humans, 3d, dragon, toy, plastic, cute, white background, simple background, colorful
low poly 3d render of a sailing boat on a calm sea with a lighthouse || no humans, 3d, low poly, sailboat, boat, ocean, lighthouse, sky
3d render of a steampunk mechanical owl with brass gears || no humans, 3d, steampunk, owl, bird, mechanical, gears, brass, realistic
3d render of a stack of colorful books with a reading lamp, cute stylized || no humans, 3d, book stack, book, lamp, colorful, stylized, simple background
3d render of an underwater scene with a tiny submarine and coral, volumetric light || no humans, 3d, underwater, submarine, coral, fish, light rays, cartoon
3d render of a cozy cabin interior with fireplace, armchair and snow outside the window || no humans, 3d, cabin, indoors, fireplace, armchair, window, snow, warm lighting
3d render of a pastel isometric game level with platforms and coins || no humans, 3d, isometric, platform, coin, pastel colors, video game, simple background
3d render of a hyperreal strawberry splashing into milk, high speed photography style || no humans, 3d, strawberry, milk, splashing, realistic, food, simple background
3d character render of a chubby penguin chef holding a ladle, pixar style || no humans, 3d, penguin, animal, chef, chef hat, holding ladle, cartoon, cute
3d render of a sci-fi corridor with glowing panels and fog || no humans, 3d, science fiction, corridor, indoors, glowing, fog, light
3d render of a paper craft style forest with layered cut paper trees || no humans, 3d, papercraft, forest, tree, layered, colorful
3d render of a cute ghost holding a pumpkin, soft lighting, autumn colors || no humans, 3d, ghost, pumpkin, holding, cute, autumn, orange theme, simple background
3d render of a bonsai tree on a stone pedestal, studio lighting, realistic || no humans, 3d, bonsai, tree, potted plant, stone, studio lighting, realistic, simple background
3d render of a toy train set running around a christmas tree with presents || no humans, 3d, toy, train, christmas tree, gift, christmas, indoors, warm lighting
"""

STYLES["graphic"] = """
pixel art of a cozy cottage in a forest clearing with a campfire, 16-bit style || no humans, pixel art, scenery, cottage, house, forest, campfire, fire, night
flat vector illustration of a mountain landscape with sun and birds, minimalist, limited palette || no humans, flat color, vector art, scenery, mountain, sun, bird, minimalism, limited palette
line art of a cat curled up asleep, clean black ink lines on white || no humans, lineart, monochrome, cat, animal, sleeping, curled up, white background, simple background
comic book style panel of a superhero woman flying over a city, halftone shading || 1girl, solo, superhero, cape, flying, city, comic, halftone, sky, bodysuit
pixel art of a spaceship flying past a ringed planet, starfield || no humans, pixel art, spacecraft, flying, planet, planetary ring, space, star (sky)
flat illustration of a woman working on a laptop in a home office with plants, corporate memphis style || 1girl, solo, flat color, laptop, computer, desk, sitting, plant, indoors, vector art
manga style black and white panel of a samurai man facing the wind, screentone || 1boy, solo, samurai, monochrome, greyscale, manga, screentone, katana, wind, comic
sticker design of a happy avocado with a white border, cute kawaii style || no humans, sticker, avocado, food, smile, cute, white border, simple background, chibi
minimalist geometric poster of a fox made of triangles, flat colors || no humans, animal, fox, geometric, triangle, flat color, minimalism, poster (medium), simple background
pixel art of a knight fighting a slime in a dungeon, retro rpg || 1boy, solo, pixel art, knight, armor, sword, slime (creature), dungeon, fighting, retro artstyle
vintage travel poster of a mountain resort with skiers, art deco style, bold colors || scenery, poster (medium), art deco, mountain, skiing, snow, retro artstyle, text-free
flat vector icon set of weather symbols: sun, cloud, rain, snow, lightning || no humans, icon, flat color, vector art, sun, cloud, rain, snowflake, lightning, white background, simple background
line art coloring page of a mandala with flowers, symmetric, black on white || no humans, lineart, mandala, flower, symmetry, monochrome, white background, simple background
pixel art of a rainy city street at night with glowing shop windows || no humans, pixel art, scenery, city, street, night, rain, window, light
comic style illustration of a detective dog in a trench coat, bold outlines || no humans, animal, dog, detective, trench coat, hat, comic, thick outlines, cartoon
flat design illustration of a coffee shop storefront with an awning, pastel palette || no humans, flat color, vector art, shop, cafe, awning, storefront, pastel colors, building
ink sketch of an old european street with cafes, loose urban sketching style || no humans, ink (medium), sketch, scenery, street, cafe, building, traditional media, monochrome
pixel art portrait of a wizard man with a long white beard and a starry hat || 1boy, solo, pixel art, wizard, old man, long beard, white beard, wizard hat, star (symbol), portrait
cartoon illustration of a happy sun and cloud holding hands, storybook style || no humans, cartoon, sun, cloud, holding hands, smile, blue sky, cute
risograph style print of a bicycle with a basket of flowers, two-color, grain texture || no humans, bicycle, basket, flower, risograph, limited palette, grain, simple background
pixel art of an underwater scene with fish, seaweed and a treasure chest || no humans, pixel art, underwater, fish, seaweed, treasure chest, bubble
line art of a hand holding a flower, single continuous line, minimalist || no humans, lineart, hand, holding flower, flower, minimalism, monochrome, white background
flat vector illustration of a rocket launching with smoke clouds, retro space poster || no humans, flat color, vector art, rocket, smoke, space, poster (medium), retro artstyle
comic book style action panel of a robot punching through a wall, speed lines, onomatopoeia-free || no humans, robot, mecha, punching, wall, debris, comic, speed lines, dynamic pose
cute chibi style illustration of a barista bear holding a coffee cup || no humans, animal, bear, chibi, barista, apron, holding cup, coffee, cute, simple background
pixel art of a farm with a red barn, crops and a windmill, sunny day || no humans, pixel art, scenery, farm, barn, crop, windmill, sky, day
stained glass style illustration of an owl with bold black leading lines || no humans, animal, owl, bird, stained glass, colorful, thick outlines
flat illustration of a bookshelf full of colorful books and plants, cozy || no humans, flat color, vector art, bookshelf, book, plant, potted plant, indoors, cozy
blueprint style technical drawing of a vintage steam locomotive, white lines on blue || no humans, blueprint, technical drawing, train, locomotive, steam locomotive, blue background, lineart
pixel art of a castle on a hill at sunset with clouds, retro game title screen || no humans, pixel art, scenery, castle, hill, sunset, cloud, retro artstyle
"""

STYLES["fantasy_scifi"] = """
concept art of a giant ancient tree city with bridges and glowing windows in a misty forest || no humans, scenery, fantasy, tree, giant tree, city, bridge, window, glowing, fog, forest
epic fantasy illustration of a dragon flying over a snowy mountain range at dawn || no humans, dragon, flying, mountain, snow, dawn, sky, cloud, fantasy, wings
sci-fi concept art of a space station orbiting a gas giant, detailed hull, stars || no humans, science fiction, space station, planet, gas giant, space, star (sky), spacecraft
concept art of a steampunk airship docked at a floating harbor above the clouds || no humans, steampunk, airship, aircraft, harbor, floating island, above clouds, cloud, sky
fantasy illustration of an elf ranger woman with long blonde hair and a green cloak in an enchanted forest || 1girl, solo, elf, pointy ears, long hair, blonde hair, green cloak, ranger, bow (weapon), forest, glowing, fantasy
sci-fi illustration of a neon cyberpunk megacity at night with flying cars || no humans, science fiction, cyberpunk, city, cityscape, night, neon lights, flying car, skyscraper
concept art of a desert caravan of camels passing ancient sandstone ruins under two suns || no humans, scenery, desert, camel, animal, caravan, ruins, sun, fantasy, sand
fantasy illustration of a wizard tower on a cliff during a thunderstorm, lightning || no humans, scenery, tower, cliff, storm, lightning, dark clouds, fantasy, castle
sci-fi concept art of a mech robot standing in a rainy industrial yard, floodlights || no humans, mecha, robot, science fiction, rain, factory, industrial, light, night
concept art of an underwater kingdom with coral palaces and mermaid statues, light rays || no humans, scenery, underwater, palace, coral, statue, light rays, fantasy, fish
fantasy illustration of a dwarf blacksmith man forging a glowing sword in a mountain hall || 1boy, solo, dwarf, beard, blacksmith, hammer, anvil, sword, glowing, fire, indoors, fantasy
sci-fi illustration of an astronaut man standing on an alien planet with giant crystals and two moons || 1boy, solo, astronaut, spacesuit, standing, alien planet, crystal, two moons, sky, science fiction
concept art of a floating castle connected by chains above a sea of clouds || no humans, scenery, castle, floating island, chain, above clouds, cloud, fantasy, sky
fantasy illustration of a phoenix rising from flames, golden and red feathers || no humans, phoenix, bird, fire, flames, wings, spread wings, glowing, fantasy
sci-fi concept art of a retro-futuristic diner on mars with a rover parked outside || no humans, science fiction, retro artstyle, diner, building, mars, desert, vehicle, red sky
fantasy illustration of a crystal cave with glowing blue mushrooms and an underground lake || no humans, scenery, cave, crystal, mushroom, glowing, lake, underground, fantasy, blue theme
concept art of a viking longship sailing through fjords under the northern lights || no humans, scenery, ship, viking, boat, fjord, aurora, night sky, mountain, ocean
sci-fi illustration of a bio-dome garden on the moon with earth visible in the sky || no humans, science fiction, dome, garden, plant, moon, earth (planet), space, scenery
fantasy illustration of a griffin perched on a castle tower at sunset || no humans, griffin, perched, castle, tower, sunset, wings, fantasy, sky
concept art of an ancient temple overgrown by jungle with a giant stone face || no humans, scenery, temple, ruins, jungle, overgrown, statue, stone, fantasy
sci-fi illustration of a woman pilot in a sleek flight suit in a starship cockpit, holographic controls || 1girl, solo, pilot, bodysuit, sitting, cockpit, spacecraft, hologram, science fiction, space
fantasy illustration of a cozy hobbit-style hill house with a round door and a garden || no humans, scenery, house, round door, hill, garden, flower, fantasy, cozy
concept art of a frozen kingdom with ice spires and a lone traveler on horseback || 1boy, solo, riding, horse, cloak, ice, snow, castle, spire, winter, fantasy, scenery
sci-fi illustration of a futuristic maglev train speeding through a glass tunnel city || no humans, science fiction, train, tunnel, glass, city, futuristic, motion blur
fantasy illustration of a giant turtle carrying a small village on its shell across the ocean || no humans, turtle, giant animal, village, house, ocean, fantasy, sky, cloud
concept art of a post-apocalyptic city reclaimed by nature, deer in the street, peaceful || no humans, scenery, post-apocalypse, ruins, city, overgrown, deer, animal, plant, street
fantasy illustration of a knight man in golden armor kneeling before a glowing sword in a stone, forest clearing || 1boy, solo, knight, armor, gold armor, kneeling, sword, sword in stone, glowing, forest, fantasy
sci-fi concept art of a huge alien jellyfish creature floating over a desert, eerie light || no humans, science fiction, alien, jellyfish, floating, desert, sky, glowing
fantasy illustration of a moonlit fairy garden with fireflies and tiny glowing houses || no humans, scenery, fairy, garden, firefly, night, moonlight, glowing, house, fantasy
concept art of a giant robot sleeping in a field of flowers, overgrown with moss || no humans, robot, giant robot, lying, flower field, moss, overgrown, science fiction, scenery
"""


def records():
    out = []
    for style, block in STYLES.items():
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        for i, ln in enumerate(lines, 1):
            prose, tags = (s.strip() for s in ln.split("||"))
            out.append({"id": f"{style}-{i:02d}", "style": style, "prose": prose, "tags": tags,
                        "source": SOURCE, "license": LICENSE})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = records()
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} prompts → {a.out}", file=sys.stderr)
