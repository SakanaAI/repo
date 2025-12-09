
PROMPT = """
Write a high-quality answer for the given question using only the provided table.

{table}

Question: {question}
Answer:
""".strip()

TABLE_TYPE = [
    "table", "tsv", "csv", "html"
]


FEWSHOT_PROMPT = """
Write a high-quality answer for the given question using only the provided table.

## Example 1:

| Municipality/Communes | Coalition for the Citizen    | Coalition for the Future  | Other/Independent            | Winning Party/Coalition   | Voter Turnout |
| Aranitas              | Robert Brahaj (42.39 %)      | Viktor Mahmutaj (57.61 %) | –                            | Coalition for the Future  | 57%           |
| Ballsh                | Dallandyshe Allkaj (47.97 %) | Ilir Çela (52.03 %)       | –                            | Coalition for the Future  | 55%           |
| Fratar                | Sabire Hoxhaj (49.75 %)      | Astrit Sejdinaj (50.25 %) | –                            | Coalition for the Future  | 57%           |
| Greshicë              | Baftjar Bakiu (55.24 %)      | Bilbil Veliaj (44.76 %)   | –                            | Coalition for the Citizen | 53%           |
| Hekal                 | Qemal Pashaj (41.99 %)       | Eqerem Beqiraj (58.01 %)  | –                            | Coalition for the Future  | 50%           |
| Kutë                  | Gentjan Dervishaj (49.70 %)  | Ramis Malaj (50.30 %)     | –                            | Coalition for the Future  | 55%           |
| Ngraçan               | Nuri Koraj (65.52 %)         | Besnik Shanaj (34.48 %)   | –                            | Coalition for the Citizen | 70%           |
| Qendër                | Agron Kapllanaj (65.45 %)    | Sybi Aliaj (34.55 %)      | –                            | Coalition for the Citizen | 57%           |
| Selitë                | Altin Bregasi (51.75 %)      | Hekuran Resulaj (45.61 %) | Nezir Jaupaj (PKSH) (2.63 %) | Coalition for the Citizen | 62%           |

Question: what municipality comes after qender?
Answer: Selitë

## Example 2:

| Years of appearance | Title                                         | Network        | Character name                                                                 | Actor                                             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 1985                | An Early Frost                                | NBC            | Michael Pierson                                                                | Aidan Quinn                                       | The first made-for-television film to address people with AIDS.                                                                                                                                                                                                                                                                                                                                                                                                             |
| 1986                | St. Elsewhere                                 | NBC            | Dr. Robert Caldwell                                                            | Mark Harmon                                       | In "Family Feud" aired Jan. 29, 1986, Dr. Caldwell was diagnosed with HIV after leaving the hospital his former colleagues are informed of his death in season six.                                                                                                                                                                                                                                                                                                         |
| 1987                | Designing Women                               | CBS            | Kendall Dobbs                                                                  | Tony Goldwyn                                      | In "Killing All the Right People", Kendall is a young gay man with AIDS who asks the women to design his funeral.                                                                                                                                                                                                                                                                                                                                                           |
| 1987                | The Equalizer                                 | CBS            | Mickey Robertson                                                               | Corey Carrier                                     | Six year old boy with AIDS is protected from harassment from his neighbors by the titular character.                                                                                                                                                                                                                                                                                                                                                                        |
| 1988                | Go Toward the Light                           | CBS            | Ben Madison                                                                    | Joshua Harris                                     | A young couple face the realities of life with their child who is diagnosed with AIDS. The young couple (Linda Hamilton, Richard Thomas) try to save their young son from the virus.                                                                                                                                                                                                                                                                                        |
| 1988                | Midnight Caller                               | NBC            | Mike Barnes Tina Cassidy Kelly West Ross Parker                                | Richard Cox Kay Lenz Julia Montgomery J. D. Lewis | In "After It Happened", Mike is a bisexual man who deliberately infects men and women, including Tina and Kelly, with HIV. Ross is Mike's former partner, who Mike abandons when Ross gets sick. The episode was controversial, drawing protests from San Francisco AIDS groups who believed the episode would encourage violence against gay people and people with AIDS. Kay Lenz won an Emmy Award for her portrayal. She reprised her role in 1989's "Someone to Love". |
| 1989–1991           | Degrassi High                                 | CBC Television | Dwayne Meyers                                                                  | Darrin Brown                                      | Heterosexual white male teenager, infected by a summer girlfriend.                                                                                                                                                                                                                                                                                                                                                                                                          |
| 1989                | The Ryan White Story                          | ABC            | Ryan White                                                                     | Lukas Haas                                        | 13-year-old haemophiliac who contracted AIDS from factor VIII                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1990–2003           | EastEnders                                    | BBC            | Mark Fowler                                                                    | Todd Carty                                        | Heterosexual male; former runaway who returned to his family after contracting HIV; died of an AIDS-related illness. He is the world's first soap opera character to contract the disease, and also the first to portray an HIV/AIDS character on a major television show outside North America.                                                                                                                                                                            |
| 1991–1993           | Life Goes On                                  | ABC            | Jessie                                                                         | Chad Lowe                                         | white male teenager, infected by girlfriend.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1991                | thirtysomething                               | ABC            | Peter Montefiore                                                               | Peter Frechette                                   | gay male, infected by his one of his partners                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1992                | Something to Live for: The Alison Gertz Story | ABC            | Alison Gertz                                                                   | Molly Ringwald                                    | female, infected during a one night affair                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 1993                | And the Band Played On                        | HBO            | Various                                                                        | Various                                           | The shows details the original discovery of AIDS and early problems in dealing with the disease                                                                                                                                                                                                                                                                                                                                                                             |
| 1993                | NYPD Blue                                     | ABC            | Ferdinand Holley                                                               | Giancarlo Esposito                                | Appears on episode titled "Holley and the Blowfish", character is a police informant who robs drug dealers. Was infected by drug usage.                                                                                                                                                                                                                                                                                                                                     |
| 1993–1995           | General Hospital                              | ABC            | Stone Cates                                                                    | Michael Sutton                                    | white male teenager                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1994                | The Real World                                | MTV            | Pedro Zamora                                                                   |                                                   | openly gay, infected by one of his partners, famous AIDS educator                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1994-               | General Hospital                              | ABC            | Robin Scorpio                                                                  | Kimberly McCullough                               | white female teenager, infected by her boyfriend Stone Cates.                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1996                | Murder One                                    | ABC            | Richard Cross                                                                  | Stanley Tucci                                     | Cross is an unscrupulous businessman whose imminent death from AIDS leads to an ethical awakening.                                                                                                                                                                                                                                                                                                                                                                          |
| 1997                | Oz                                            | HBO            | Various including Antonio Nappa, James Robson, Nat Ginzburg and Clarence Seroy | Various                                           | Men, infected while in prison, HIV positive inmates are isolated in Unit E, a cell block exclusively for HIV positive inmates not allowed to be amongst general inmate population.                                                                                                                                                                                                                                                                                          |
| 1998                | Law & Order                                   | NBC            | Kenneth "Twist" Stark                                                          | Jason Hayes                                       | In the 1998 episode "Carrier", Stark was charged with murder and attempted murder for deliberately infecting women with HIV.                                                                                                                                                                                                                                                                                                                                                |
| 1999                | ER                                            | NBC            | Jeanie Boulet                                                                  | Gloria Reuben                                     | African-American female adult, infected by husband.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2001–2005           | Queer as Folk                                 | Showtime       | Vic Grassi Ben Bruckner James "Hunter" Montgomery                              | Jack Wetherall Robert Gant Harris Allan           | Vic was Michael Novotney's uncle. Ben was Michael's boyfriend and a college professor. Hunter was a former hustler who became Ben and Michael's foster son.                                                                                                                                                                                                                                                                                                                 |
| 2002                | Takalani Sesame                               | SABC           | Kami                                                                           | puppet                                            | Five-year-old female puppet. Contracted HIV via tainted blood transfusion. World's first HIV-positive Sesame Street Muppet.                                                                                                                                                                                                                                                                                                                                                 |
| 2005                | Nip/Tuck                                      | FX             | Gina Russo                                                                     | Jessalyn Gilsig                                   | Sex addict. Had a child before she knew she was infected with the virus but the baby did not contract it. Re-appeared briefly to work as a receptionist at McNamara/Troy but died soon after when she fell off a building.                                                                                                                                                                                                                                                  |
| 2005–2008           | Home and Away                                 | Seven Network  | Cassie Turner                                                                  | Sharni Vinson                                     | Cassie contracted HIV after sleeping with her older boyfriend Henk, who had contracted the disease from a drug addicted former girlfriend.                                                                                                                                                                                                                                                                                                                                  |
| 2007 -              | The Best Years                                | The N (U.S.)   | Lee Campbell                                                                   | Alan Van Sprang                                   | Bisexual Lee owns local hot spot nightclub Colony.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2008                | Hollyoaks                                     | Channel 4      | Malachy Fisher                                                                 | Glen Wallace                                      | Heterosexual male; contracted the disease and kept it a secret from girlfriend Mercedes. He told his brother Kris as he may have contracted it from a one night stand with Merdedes Mercedes & Malachy were to marry but ended their relationship, a row in the pub had Mercedes revealing his disease to his friends & Mum, They couple later married, Mercedes is awiting her results.                                                                                    |
| 2008                | South Park                                    | Comedy Central | Eric Cartman Kyle Broflovski                                                   | Trey Parker Matt Stone                            | In the episode 'Tonsil Trouble' Cartman is infected with HIV during a tonsillectomy When Broflovski mocks him, Cartman secretly injects him with infected blood to pass on the virus. Both are subsequently cured through injections of money into the bloodstream.                                                                                                                                                                                                         |

Question: collectively, how many shows did hbo and mtv air?
Answer: 3

## Example 3:

| Name                    | Date of birth     | Club              | Division                | Appointed         | Time as manager   |
| Francis Bosschaerts     | 15 October 1956   | Heist             | Belgian Second Division | 1 June 1999       | 15 years, 25 days |
| Peter Maes              | 1 June 1964       | Lokeren           | Belgian Pro League      | 20 May 2010       | 4 years, 37 days  |
| Hein Vanhaezebrouck     | 16 February 1964  | Kortrijk          | Belgian Pro League      | 6 June 2010       | 4 years, 20 days  |
| Frederik Vanderbiest    | 10 October 1977   | Oostende          | Belgian Pro League      | 15 February 2011  | 3 years, 131 days |
| Arnauld Mercier         | 4 June 1972       | Boussu Dour       | Belgian Second Division | 21 April 2011     | 3 years, 66 days  |
| Frank Defays            | 23 January 1974   | Virton            | Belgian Second Division | 6 June 2011       | 3 years, 20 days  |
| Serhiy Serebrennikov    | 1 September 1976  | Roeselare         | Belgian Second Division | 30 June 2011      | 2 years, 361 days |
| Regi Van Acker          | 25 April 1955     | Hoogstraten       | Belgian Second Division | 23 November 2011  | 2 years, 215 days |
| Francky Dury            | 11 October 1957   | Zulte Waregem     | Belgian Pro League      | 30 December 2011  | 2 years, 178 days |
| Dante Brogno            | 2 May 1966        | Tubize            | Belgian Second Division | 26 February 2012  | 2 years, 120 days |
| Eric Franken            |                   | ASV Geel          | Belgian Second Division | 20 March 2012     | 2 years, 98 days  |
| John van den Brom       | 4 October 1966    | Anderlecht        | Belgian Pro League      | 30 May 2012       | 2 years, 27 days  |
| Tintín Márquez          | 7 January 1962    | Eupen             | Belgian Second Division | 6 July 2012       | 1 year, 355 days  |
| Lorenzo Staelens        | 30 April 1964     | Cercle Brugge     | Belgian Pro League      | 2 April 2013      | 1 year, 85 days   |
| Dennis van Wijk         | 16 December 1962  | Westerlo          | Belgian Second Division | 29 April 2013     | 1 year, 58 days   |
| Stanley Menzo           | 15 October 1963   | Lierse            | Belgian Pro League      | 14 May 2013       | 1 year, 43 days   |
| Yannick Ferrera         | 24 September 1980 | Sint-Truiden      | Belgian Second Division | 24 May 2013       | 1 year, 33 days   |
| Guy Luzon               | 7 August 1975     | Standard Liège    | Belgian Pro League      | 27 May 2013       | 1 year, 30 days   |
| Jimmy Floyd Hasselbaink | 27 March 1972     | Antwerp           | Belgian Second Division | 29 May 2013       | 1 year, 28 days   |
| Philippe Médery         |                   | Visé              | Belgian Second Division | 31 May 2013       | 1 year, 26 days   |
| Felice Mazzu            | 12 March 1966     | Charleroi         | Belgian Pro League      | 1 June 2013       | 1 year, 25 days   |
| Stijn Vreven            | 18 July 1973      | Lommel United     | Belgian Second Division | 1 June 2013       | 1 year, 25 days   |
| Michel Preud'homme      | 24 January 1959   | Club Brugge       | Belgian Pro League      | 21 September 2013 | 0 years, 278 days |
| Lionel Bah              | 2 February 1980   | WS Brussels       | Belgian Second Division | 21 September 2013 | 0 years, 278 days |
| Guido Brepoels          | 7 June 1961       | Dessel Sport      | Belgian Second Division | 24 September 2013 | 0 years, 275 days |
| Čedomir Janevski        | 3 July 1961       | Mons              | Belgian Pro League      | 27 September 2013 | 0 years, 272 days |
| Mircea Rednic           | 9 April 1962      | Gent              | Belgian Pro League      | 1 October 2013    | 0 years, 268 days |
| Bob Peeters             | 10 January 1974   | Waasland-Beveren  | Belgian Pro League      | 5 November 2013   | 0 years, 233 days |
| Rachid Chihab           |                   | Mouscron-Péruwelz | Belgian Second Division | 19 December 2013  | 0 years, 189 days |
| Franky Vercauteren      | 28 October 1956   | Mechelen          | Belgian Pro League      | 5 January 2014    | 0 years, 172 days |
| Jean-Guy Wallemme       | 10 August 1967    | RWDM Brussels     | Belgian Second Division | 30 January 2014   | 0 years, 147 days |
| René Desaeyere          | 14 September 1947 | Aalst             | Belgian Second Division | 5 February 2014   | 0 years, 141 days |
| Emilio Ferrera          | 19 June 1967      | Genk              | Belgian Pro League      | 24 February 2014  | 0 years, 122 days |
| Ivan Leko               | 7 February 1978   | OH Leuven         | Belgian Pro League      | 25 February 2014  | 0 years, 121 days |

Question: how many total managers has there been?
Answer: 34

## Example 4:

| Release date   | Album                                              | Record label      | UK Albums Chart | U.S. Billboard 200 Chart |
| July 1983      | The Alarm (EP)                                     | I.R.S. Records    | -               | 126                      |
| February 1984  | Declaration                                        | I.R.S. Records    | 6               | 50                       |
| October 1985   | Strength                                           | I.R.S. Records    | 18              | 39                       |
| November 1987  | Eye of the Hurricane                               | I.R.S. Records    | 23              | 77                       |
| November 1988  | Electric Folklore Live                             | I.R.S. Records    | 62              | 167                      |
| 1988           | Compact Hits                                       | A&M Records       | -               | -                        |
| September 1989 | Change ¥                                           | I.R.S. Records    | 13              | 75                       |
| November 1990  | Standards                                          | I.R.S. Records    | 47              | 177                      |
| April 1991     | Raw ¥¥                                             | I.R.S. Records    | 33              | 161                      |
| 2001           | Eponymous 1981-1983 ¢                              | 21st Century      | -               | -                        |
| 2001           | Declaration 1984-1985 ¢                            | 21st Century      | -               | -                        |
| 2001           | Strength 1985-1986 ¢                               | 21st Century      | -               | -                        |
| 2001           | Eye of the Hurricane 1987-1988 ¢                   | 21st Century      | -               | -                        |
| 2001           | Electric Folklore Live 1987-1988 ¢                 | 21st Century      | -               | -                        |
| 2001           | Change 1989-1990 ¢                                 | 21st Century      | -               | -                        |
| 2001           | Raw 1990-1991 ¢                                    | 21st Century      | -               | -                        |
| 21 Sept 2002   | Close≠                                             | 21st Century      | -               | -                        |
| October 2002   | The Normal Rules Do Not Apply≠                     | 21st Century      | -               | -                        |
| 17 Dec 2002    | Trafficking≠                                       | 21st Century      | -               | -                        |
| 17 Dec 2002    | Edward Henry Street≠                               | 21st Century      | -               | -                        |
| January 2003   | Coming Home≠                                       | 21st Century      | -               | -                        |
| 15 Jan 2003    | Live at Hammersmith Palais 1984                    | 21st Century      | -               | -                        |
| 23 June 2003   | The Alarm EP - 20th Anniversary Collectors Edition | 21st Century      | -               | -                        |
| 17 Jul 2003    | Live at Glasgow Garage =                           | 21st Century      | -               | -                        |
| 17 Jul 2003    | Live at Liverpool Masque Theatre =                 | 21st Century      | -               | -                        |
| 17 Jul 2003    | Live at London Mean Fiddler=                       | 21st Century      | -               | -                        |
| 19 Oct 2003    | The Sound and the Fury =                           | Shakedown Records | -               | -                        |
| 2004           | In the Poppyfields ¶                               | Snapper Music     | 107             | -                        |
| 2004           | Live In the Poppyfields ¶                          | Snapper Music     | -               | -                        |
| 2005           | Alt-Strength                                       | 21st Century      | -               | -                        |
| 2006           | Under Attack $                                     | Liberty           | 138             | -                        |
| 2006           | The Best of The Alarm and Mike Peters              | EMI               | -               | -                        |
| 2006           | Alarm MMV - The Saturday Gigs $                    | 21st Century      | -               | -                        |
| 2007           | The Collection ^                                   | EMI Gold          | -               | -                        |
| July 2007      | Three Sevens Clash ฿                               | 21st Century      | -               | -                        |
| August 2007    | Fightback ฿                                        | 21st Century      | -               | -                        |
| September 2007 | This is not a Test ฿                               | 21st Century      | -               | -                        |
| October 2007   | Situation Under Control ฿                          | 21st Century      | -               | -                        |
| November 2007  | Call to Action ฿                                   | 21st Century      | -               | -                        |
| December 2007  | 1983/84 ฿                                          | 21st Century      | -               | -                        |
| January 2008   | Counter Attack ฿                                   | 21st Century      | -               | -                        |
| 2008           | Guerilla Tactics ¤                                 | 21st Century      | -               | -                        |
| 2008           | The Alarm - BBC Radio Sessions 1983-1991           | 21st Century      | -               | -                        |
| April 2010     | Direct Action °                                    | 21st Century      | -               | -                        |
| March 2013     | Vinyl (2012 film) Soundtrack °                     | -                 | -               |                          |

Question: what was the first album released?
Answer: The Alarm (EP)

## Example 5:

{table}

Question: {question}
Answer:
""".strip()