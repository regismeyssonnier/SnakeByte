#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("movbe,avx,avx2,fma,sse4.2,popcnt,bmi,bmi2,lzcnt")  //Enable AVX
#include <x86intrin.h> //AVX/SSE Extensions
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <math.h>
#include <ctime>
#include <chrono>
#include <map>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <stack>
#include <cstring>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <cassert>
#include <cmath>
using namespace std::chrono;
using namespace std;



wstring model_0_weight = L"愬嶬攟懌戂愰懴慳忛憎彯循恎惇䮔悁慇幭愲彄怏縵慼憗徛慂崀斒懷扚憵帇恓嵨楌岐彇徉悇忱恲怗彽庅忏噌庪廦扜恸抾捓姄渞幔撐朑忀巆幵愂徒段怸彻彻憎庵弦普幢徟忀峘怊怌思怫巧悩嶼懸弎愰岅帥澘慼恘廚愉怾忆盈怾惸忰峌栀廂懇崘崾嶳弢洫戊崉懠庮檳弝悀彯彛悩幪守憖怅惫挍惒扷徤憗枀憅戾徒慓怕恫忭攔憠忧恐库弱庱穢庤恤彸手䨌嵅壈敧拋崛妑捋弌怗彦怰埿幡憛愳徊庩徟揖憈御拶忯彿戟捞嬻島幓惲廙弬懿愫悌楛悿愿怒怦愌弘Ⲕ彿強庢彅廽摐廚憾庘惜忂惮巣烎弪往撒徏悪幡怠恌彂摭惿弨摰搓嬆旗濲彍抋帢斣掅敏按怡怫慯庻慢愒弈彠幯椺愣怮島戌婧嶰方幱摮徝揈彛嚵柡惥忢怣弄彩憆恡徇弒樦愷店怌幤怕庄怊慘恾恔怖康怴怼弆彄彗惯恡征悟庬很憐弡忲抓慰抑尋按惲法投敗彼恱扗懑忘焾愋悫彊惷幸廾以庪廃彗戆弋弖彩愬庖幖囚慧啥廣弮慝墫度慢弜慆悩悰幰慣愵廀忰堹憝搦捦悆嵼廩崍幧夝感怚巕忨愞悔怭慂慊掽庇惤撉懎灶嵽愖槧嬪惐憎幛嫗岷悹悠剂愌弡慓念庹幵埬慷弅摇娖戺捲拜彞悝拡憋扺暡揸懟帔檨惉愣忷廷悕悮嶩怺惚嬵忢忡柧擛故恾悈攆壛幧崳揯擾孤慇徳弚忙平徒屼慐悏帐怖摡寬拡廮掯挎倵忨憽愣彄想果彆憖彵彣慩怊䪆愶怤愭嵴掣帠浹孺属掴揱掭彺忐懄惲奮慞态恏惥慩悔嶺慒忊懧摢掙旰慓恱揤屢戲堒塧怪応庥星恣平恚恳怭怴晃庰弗惘忔䭅娧稈怩崎忥怭怟怩扐总怟掚弮愼廞庵悲慱嶔悀惓屴崙挙堓夅幽屝恛恠敊惵明忟惼撀怍录废恙怛悌狨慼怵悔忀嶶廩度扠投忠忶挻玸揞惥弴凈弪徵怔庹惒庢昇忸忀彲怹徂庂慵寄嗾烱挶悧旑嵐廃彦孁徑廂徲弨幧廫孢彰弡忕擑帉強果巼敂慫悝懘桒桄挔廂墅态廑惡忟忁悙猿彬弮忚搛思憢慇弛志懃忀恋庑怂恚恘撲患廈忑悾惘异芏憅惄姧愂搑洺捩櫲嵜愦嶋拋愉戶崽孏擒庩惐忑惩恲徛挄惜憜彰彾彜樰攥忏樽戻懄摽嗷媼弤恨扥庖彌忸廃徰惊威惋弮帐旕崬悀恺扟庡拃嵤悾彧廋嵘德呋惫怵彚愲愒意㼵彔廟庋弓妟昂娵拻怲攝審敯嵝戳恫帯儊惨惜弑弜怅廀懓廫徯揤描惨庎拡左嵿恕嶩愮憵搗廫悁夕徒悊廕従庿惸癒彼怘";

wstring model_0_bias = L"懟挘恺巒扎嵒彲悐忹徍扩廌帹报掲幸戕戢慩掬崲微批慮惙彃峖慁怒惒接屢";

wstring model_2_weight = L"慈拝懩末嬇姡庹揙嚞媔弪徖嚏摳引彃浀廃埤媺宛徧帕寀嫄揧檞帰彡恖怀挷愴嗠抪差抝彐庎哟庲忷握庼弭挿嶱夓戎憕幔弣恌悏憫垳拨塼憪晱庯杙揝州挲抔悟撿壛摳媼刜壕怊從怳娅杰座帔戩嬸呈掷変崶彩屠媍攒崺忋寔敍憶徊戓廃戭栧崼悌尰擠埻搒廾徤囘惻扖彣拞怄扃惵嵱壐愼拿媛暛托愱娃昛弘憊州怟夘嗃户晳晾杪摇摺宠廱怓姛廩暩弰挟杮悁惛戟拲擮挺婲摯曗憌夑夝妚楼忑廄壬尗撤憓彗怙愵巂慅奘峷孷扇揮惿憉恽敶悲昲撿憐戲捃扦戧岙扥岲弈惍廰挡嶱悁巷殀廘愛愛彞尊悥或墬怇擔帟抽廀婎朴恽愇帧想嶛峉杩庢恻咎捕弧庹慷嫀拷柤惏彜憴弲捽崅慇孶幵曌揥慑挟尸恪怡搈峤廆斉慶廝寃延昦抩悩搀幯恑嵛巶廟悑抬悶帣愤搝宛掃挘嶺戸恊壓敚抸彺徳旴旳忕搼扺揕掑弭嬊恹彡愥幕恙嬯斠属従控惸屺晁徻摎摫有彾寁揜川婥帶戎歵峏抂姛惾徼摾媍捍愩攂慽拶揫嶔彜廼弛岍捿懋憄慝愾彅尸恚己扄弗屔墜慬攪栶廕壔攏悺德折帑懄库嵰庁彞掌感廓悏揌孇憸挧帍懑廟寵拦抆愩左晏捎庐擣拂掖愶慷嚍坥巬奺媢嵍怂彾廕庉忁憤抐庄孓嶎寺憩忙宩搜垘惥栈悍岵拓媝栔梶握拨弯慹慶掍扨櫲拮杊埾徉廏屁孀挆巏戨抛岅塩床挲搶惘挎巣撪寢忑幎尪庰恥悃廋悞很庶廽恗弭忺愶怜怹弻得徙廦悗廼怌悚廹徟庰弴惶弘怫弦彺徐恐嵜掬扳嫾墔屍汤嵧宏弋弰敭慃悿屮家忘廦持峏巧昢恰嬈怉呚悈嶀榱懶崺";

wstring model_2_bias = L"廘戥悑廥寺嶇悆揬憨帮摪悼彑徏廷拚";

wstring model_4_weight = L"峝巺揳憘拖幂愉巑戕帾庭惴巼帣愖嵇";

wstring model_4_bias = L"搙";

//PUCT
wstring net_0_weight_pol = L"搚尙抷熷囖敁判悵椽對暼敨屿徨峅弮怣慚幧庑张屉弆廡椠彛媘孪屄岄層寂摈媒楐惮恺梑沓慵慹恑彩愝憚泟惽慍愬廕就濄攭懀慞捁愕圁娨忀摊戝媷愑惜彳惱廾怔噈弻広戰䣟娖夘峍朐扃悪惜寝橻擌憇撅欭忮弓幱役徝忼艕愅彔嫉恥搴摁媲撑奰嘡嚭晝咼早廛忒撉年徴恪憑悴廏共彄慷栟属吳独凣孰売嶻弨惟宪宴扛廒啫憉忧庝悦彐御姸惽引峣嵰曯梾淴摣杼宄怋勯奒掺忮憾焐怾怋弆弡必弦榭怵急曰役彤愇搦剀拷愍曢嶕婰攙摰失晅惕彎惙慂恬廚嶫廠恆嫧敍忘擊弐堀曞捛斓奩恄宀怴岑曯建慖悿彣悡恑婄憋您忔意召囷叒叉姩孹攫攒噾嬗晰宓溕弗弅忥庵徻忢汖愽弎戉ᡂ幬幉寯扔拾棪掩扢摍扰嵴廏嚦愽弝悼幬归彻截恸恨妏延巓嵏愫慑悰扤攀憋朙斈晵柁湋忤怾幤廜恧忯浽憎惭攨岽橕宆峻帘扆挓怇屟呸姽惈廢噳廈庒徧平悬忺剂弆徝拯幭愉戼僻凌斢扮劆幸帯寫樥愛䧙悯惤庸庱忠弰拑恃志慸忝僔䵙墕桃峑斱夶婎悕懠攧捴斬恰廟怰広惧悺槎徚徛揳掠峊嶢嬒嵯嫣弶捛栧洱嫕恃懕姌愔弭愳愘彳慾䵩徿慟崄懧婩桃昫枢怋噹曔摱妤帿嚰帣悋惊愵悩張弢念姌悁弰抿拇䝀勄幋外惓栖揚旤扡惿惱扳檇庅廼廙廑弶惶廋廼彤昹擂朂勽斊忓夒慔敭剥擶愄徔拽槚愛忉後怆恠徖搞徣彪扆崒奜墄䵸媶攈摌嶛橼搠染幐崽惾憖彵怘廟怓恤勵廈悅撪屮搆恨慷杭旄湸拶梧惩姝懟彼岦录悶愱惓恎憐楒忷廔懴拂唶檉憣撠揰挳朓暶檑挧愙征偈怘悊徶幽忀怦廩忶弲吗巅泠堁愽妓嶐揠扄嬱潭嫲晻晶卡慺惶徣慗從愀嶔廿慇按恳柭喨揭僉巖恽擒宙楸掾崽孮挨恣度彆忰恲庬欄悡慯崋慘濥囎戂嬙憓婷毸慯寂桫孈憿嗧愷廇庯忺彉悰奴弖庹抷慖濖䡅惙媄壍撎晍柄擪报敉嵹嵟怟恣惈愝彐庶嗎慴愕恹容扲父昳展恗嘟崆搘挘屄晦憩湰彞庳很愑悂恑擓忛往愬怯ࢾ庐慁廵徭慩戂急彂嶼愩徣抜恨悃慻彚庣愑帋惥慚圊戹浾䲩捘嵻榚慪憂徴嬔拿惮憚懪弶彺恾幠庅庎昨御弔揇彡恙楷寰峲墻抏䯄檅垾探彾徜呆惈忒廫弶憈愈佂弚怊敎庅時徂棟乀嘓协壕廭椵嵼拨梘奺廒廻怃彌憄悖悾庡惜嬢情略嫑揥掐弓搿墩憹廪曜拥尓拇忠慁廁慭忖彌嶊庺後庐拒癵唝揆囓榺巛序敕怮宅惼岨止忸態弟慏徏恭嵒愽応序挠娰⃳己斬慠彦捙峛幂懛怂懃桬廾慬念庪徰徾廣怴徭學挈圐㶓尨挽廆愼家慆撼圭従慽厂弉恈徨恇忝怈撦慫惙慠懖㱆攋屸唪丮垀朼動屵忺宫幼嘆憗庆愐悎恳径斧恮序嬳徥姝戥埥既恡浬淉政嬵岖彔崄春廮悏息愊惖忌烵庶慡橔恌栊䞍廻應斻悢冈徲巩嵎扡弒染彸愇憑愘憖悈図徣慁情枝抪庻尠星戯嶃慮拮挝挊朆愕抄循庽慘慼徆彘堩徼彺患幱嶀岹凙妓妙団孂孊嚏岐宻妱呍恬庣廄彋廤慉俼彨忔寒摻䭕孰崙圀峔怠橉摠扶娈夎斧侸庫惍悢廑憝怹忺愂弢榓少埔棯抜呛榐弗喲廵怦弝擆忆姲惡庣弹廣怛徔例彾彀嶩䭶愆惡曀惍檻枭恒昶惶弭慝岽榦彼恌惤慞悼庉猹復悲昽嬥崑恬弘弬扸椤忁孮洖墦慗忞擏庁惨建庢愴您揯恛弦懟棣殧兢抓扑恁廆徰擹旐岀壍搔嶼惿徕惡慲憐悊晢急幡朴慂愡啈弆堭屑庘挕埯晧悝把泆嶨悟愈庋忮慤怬囼弭惁崓时搉幔怳屵憥抑捅姃斟彀峒巙氫徐恤恦忨恹徟徢彘弱杒左怭劐廫暀擛扥廗崾斩惾塠媘悦惖悭憈彂態徨狌憆惖朚帯奎敲嬂悹廡嫷図攒崍愵嵌掷⩽惙彞愐怲慜悙扞廪惉拣帻沺僊嚫堅挧挾枆寿徕捬娠峦桾忓得当恊恨徽壴忞憌扫嫅灕抎楼搗婩嬺愦挤尳嵦弜捙瓈恁征彖恏愶愅嚊悩惯嚪橜䲦换䚎懥属栦抿惴娩捼嵻斘䨷庡慞徤惩徹悊煗忎怶洉娿呺彔杴彘换婼杒帵恬旃嬅宧沈庬恎怃愀惨忯厧惗庳梥捀奂槮庽怬徟楖憀嶙孰庥持愯幫徲惩憎廧惓怨暻惱廣孖憳塽此岽崩椉歶杓慶壓挐嵤店弎憝怈徬意庄幽瑆怼慿塦揿崬孜择棆巗昘䩱妩柡層廹墈暦悇廀恐庶忍愢朑弐徽撡抯䊣唀崀慯柗夤媛撻宭嶰拑戈咻惒怮怜慤徏憂殂惈恀噢彰湚夗弅屲撏凐挢忤悆孤捺惉嘒恾恸徧庞恁愠嶐弔恬抛彌昳巫彞揗寕愯勪崝栥儙夻嬼渄悌憊怮慮庫慛斤慨悏峊懂獳㗥彂從庄成平峃怪婿帄慑嬷恜悓彭恉怫廷熥愂彫帒怍啣彚彥抃屌埆吔庘璓侢弿弓奙惤慮异憙彼彊嘈彚徆怷怣屷䖳求岪崥廇抧州嵪坎搕堲慲慤彤往忒彐愈榓慛慆惀恌㸰㢆慪拶廊慓將徨搷彩徝必恇悁得憋慘怎悞拞憙忽抓悽噜擵崣喲愷捣啼榇樦园怬框婘愀徕忬徸惮彛掙廓弈";

wstring net_0_bias_pol = L"想嶑恖恆峱嵥恹彊憓愐崸徚撄巆振愬曪怔总彴帛廳恗岴摤忿戛惔抱怟岦憲導思揌捹戌录巁搑愒嶙懍惫嵗徕挴拈幵料恝嵔寯岴忢奐峟懯岕愑墐悶忋怈";

wstring net_2_weight_pol = L"㢺嬴䣘袒但纩瀋岫専式怳姈巐彋昮巖喫彬掜筛揉䛲岈攂摃挗㭕劷拄䶢揧忇崵晵抺咝峭擲岆屼忯击氦惿帐屛嵲斯澤拆哳ᴜ曀哦奢庩曅懧媺昭䱗慖徣搮序彤庆抭榫栨宔展忲嶺帲岢妆嵯徺惫惤扏搔挌挄娩涿呋氱必巘嗚憰弨惆搣恏昬妸態宦咊择哓嚇磿明敹檷恛廹勯扰奭徊欅礤接媕吹戦壵滣测条杈抐廘庇嵉摯柴撏懤旎弜怗慺朮怂捝硆崣憖椅喱暅崇揫晷揩埨卙寧庙廿弾市婯媺巬䠈ኸ忦曃勸怑慺弶涗恪枯暻斑庠扶旘廖家堗永撊掅妩嵕檉憲孤㙢揓屛恗恔弓快棅挤峇慖焴䤤庇堙呖懩嫧干怗急庈擈昭挵嵶氰廜埯汉懊彷嘿擀圕䮁⭁掖怜攔昴梤孰卡岗慏忒攈嶹摆汄勉挢滜嚬孞圯形憫檚摛䃚愻桨㣉㮨掔摏潍墩䩡烥⼣哰峻戅奭弟嫟媩掰峏棢崂憎帻捺橍嚚怹慑挢欌楠呷嘩暟摦嬾怿惾敹恷傶忂杢挽扙悙棇就愚抾撏帮岳㸩憨掰劮枒䜕勍嫒彼廦慣曱倓慉搻榙撚孷悮孚掅憷擵嚌引媟性揪河朿平斂搟悠敓孼昖悻摂斤夠娓朜昱峚橢榯怓䴶寠枔慧搻宔掠氂毲岭摏宫咏嬂搢氊感䨋惡期掆捛椬峥怅嗽初嶂䳓探愘忬挮憖捭巪崬悷挃彿憯揘斳崶敝峑柑嵢嶑嵵仐娣搯抩屬掤掠弰帣灟廨庌尉朔弲婸欥戼尦嘞堰攎掑摇晞廩暑屾婀墂撉接攭曖摑廃彨晛灆橆岌彛徴坕潭恿更往曏忈槳暢摎娺埑暉昖怂恄壼孉拾埳异揰彍巏忞扩息掠殅徶檤揃憏拎惍恦溵服撃廲拟嬗崻届探械岠惾尴圞恿摩汙徠抆巩崜媭嚆廀慤屩涓巋朻更寇帀彾娟忬䶓䬱扝懯戳婚椫彝憓垆欟憕戗琉床号宍搝櫱呉怿慾弚䫄嵘挚氅持捥刲忪狜挷梂技暃攅垙拗廥庑圗嶃徲槆怦徚嘞噥曊揆橢岦擈凨帜箮塽㥞尲墨嚔媔摉搟嵍攡憁拣掞槀揖彝恬搽栊抃斁忮捆捭懶拡搡幑扣忬榎攕懂汃抡東啲捿嫩堿忎曶榋湅兓搌垄岭屴斜杧嶏幡拣欘尛巡帢哣抉悻婷嬝椿扣偁䤎搗揈擏嗇憐幂岿堛嫭巀皺娘慡曭栩怉掝揹晪揰愸惝嬩惛捲掻役岈搪捪悟塠寠柭愯幹抄斥婮暢悬氦摏塧暆擗堦怢懽廄嬾垂嚞峘橐抾愴彩撸敧堤抋暌抬妷懧圕惉厛圫樮擏悂嗀湢帪捛愮昇摽挊旆尲墻嘟搿慜囩嫸尹惷吣它栴惍抮昰倉帅旋旋摠攐嵈攬叚墸挰対憙懊挾囏嶷怘扗寥摑撺姖怉椙志包筴暦䙂弈咙挴娎攌楖摽呂尠已恧凬斬忛壦峋希崑拹挜敌樰呟慉嶯岍昈廞栞榸悦搱娴懍栾撐惈夎徐媢崇少囋暎壠宕愪廉旹檽欔咜恔娖戫傤忒掟堛啡晄杠攂廊崋庾建息廮悟徖忠忦廃徘悑怫廘恢忯悷忩弆怐彇弄忑惶忔弓彑张徲弑忙徑形忩弻怖徿恙忲徜恟徝廫徐忙惐廣廋悖恿忌恜怟惤忊悉快彵悅忸彸廧彯廩忨彇峃旋憳擷悵孒斿悇弥众摘孪宾录凨斍婤权撀忨慰䈻惰怞已怽嶩䍀斛捖捦䮎敯孻嶇慑柣挵冴懗悸卡姐檸叏悐娍寒徾惉峧攳拳摕怾䯠怼忝嶲䪘栱埶庒ⷭ拘宫傤呎膡圤嶝安悓渐海戠圢惈搁嬇嫯憕披濳冷拦埅栦徇呓嵽峡佑嫎櫪挥扽拯扅嬑楋妑徕奿廘帼洲扃孛徶搫塜掣㽫尟插恾乻兂披巻尩挡揿悈撘斳⬀憋哔䃈婌坟䫲微廎彳嶊弉愙娉嶫揷寏扲戄撖烷囐朁壵檼嵮屾慼懐岕悙智枌朰搰抆夑椯心彥圍棖圖斎扞妳憐尒怙徣懣斤妠戦堣㰩懬挴奡昡歧巓撥披彬梖毌櫪汅帨䬂怱昁幆拜扬妓廁恚屼攎捾干敮弶抛墎愠妐岁垿姌婤常报惠枏晶崮姪塓懦廵慷椟敝慤嵕摡夾桩嬔怕希戱懅巏斍堜崭孏挝柷擀效敜惟撣帝棛桶噪樂擦嫡姲恗悽尖䮅得岛巹寙儝揼岻橏巧喵崒岶抅暡抴懆抸揔忢楊抓摱悿搖愀峙撉嬗啧惋亍桬怍搤恒挐掀屌恍扁欨恂媟惥标杳擀懿慟斻忰朰塾彨峽桪崬塟愾娛巘扉璷欝嵤妇愆崤徯柌幓欱敢忬桨掝堠挽弳浗所奼曦埉掀捱愕摖僫掺拢晻涗媔孍憌悏墖怬擒抜掁抩潦䮰拊呓匐敾婀庸悮埥弻嶛宦拳擯姂欁掇壧挟搔挨弩吐帊懫徎朝巀惶悁征峐幙彶曪唹橛攞怆䴞埔寋惠槷夗屭征摡岖把抷妳挕椰崳槄湾帍宧媟晑徉撺斤槎捂密庱愈攗炒孹痬报哠旪忲恜弥徨弢悹忆彐异徍恎性徫悐开恥怙怣彺忤性恰徎忖忴延怓恟彥彏弱怸彸徃忎弍強态廧心怊廥恆彘弫弥忺悘悅怹彲悇怜役怜悖弬悹悪悹弫徹徏彨怰恳彲弅徽弗恞弳忸彰忂彩怿忶徫廟弳怍廝悿彂弨彧恳強忩心怕弍怤恤彡怬廤廫悥弻强怦忲怷廐怲恌忯庪彁恞怈当徥従恌悕忯恏怸忒彑恮忇庥庬忢柳夗搡折慇嗖抋徉徐楇䪑庳嬘常护幗揱尕墥厲尮憥憸惩帳惩愝䭱捎员喕嶉將旐掹毆垨掫捠嚻搜旝旮拍楾拚峢晈䆣宦搃渂愡拓慪楦刺廌悪嚛匏憑䄜愵悆樂娙姥僥庿懦慣忾旗枒廒撕帧欭恨捑惹愰啖寰懤娅槳帚尅嵅堊尴旧圠岻悶啦楚夓宅揶徼怷勊复挸收榇橴拴棫撇揩搋廂悮怖昪杫垉戇嗯層亳墊咯曅巩扼掯庇昃戥戽杮差摧姡峱垭䇻敻奢巗挾廏愮巿得浟攭咘䴯挝歾侯役弰娍炈廞懇揮悾剃悑姱御孴戝家扥惑拕墮惱晇扙擗揉扜嶲榜汰亰䮷挡㾳席斒搙捷昱慪敡悙捊揌揞広瘥磠廂捵怴庩尘旸嚈䘽曚栥暪嚇嫱旍䱌曓扳付慥喏娦唕搜梦堭帒尸屑据彶徧搒攁欓婒屢囂扥埞檞拗弑抾弦憹枕卆叄ᤕ䚏暆憷嵱䫼曵婏厶懙㵚悚拒晷峞儎姇捳峦杅扬汞夓嵖戸寄峈棉娵晔愐槹㪝恽因友揲彫岟嵊嵐摫杪旎弈惗氿巾愐嵞戊慠庫眊撿呟拪巐挄星司侼峭擣巓批慗怤榝怫弽廾惉彴忩怅弲忣恣忏怫弟彘彝廓彉徿怱弡怾忖怺忮忖彃悭悻怤悝弗弦彦彚彛廪廔廑恿弹徉廷彋忩恵悥恷弈忭彍廦弛弘廬庛德忉恤悔怦恩廜忶忤庵廖恰怀彀忑弉惀弜彄忓忉弰怀恶弞怽廏恣徼徰怂恩廬恪弤彽怓廳彳彻恀弅応性弶弙忯弧彈徆怽悽忬恸忕徊徻恘忌怺恨悈復怽恗彋悦忄恸当彊彼恐損撧憖惼忔擗拓堑垸廾刜奎奮櫉帩慾懖悁捻戾挬扷填愣挬憎恁徵寂患嶁剅巄攏掋孱圔慩恫愅崇嶭䘏姼拵檧恓徟愮孩怩槁櫹愦娏屉弰揓帥挵匇怺怄戁悭揪斐嚻䴎拾捂彁娨捡㴳嵢幤奘掦懒嫝宫從喐攚惧晟婠涇搬壦䨤嶄娝斏旹捡氢彰弶机掐憫宇塥殙堊擀摱姃崐堎弸扺嫻嚐歍敹塀䮽楝摤境枭珂慫欞暲";

wstring net_2_bias_pol = L"婏擙揳帮怌攥悚岥岙捚宵撜斟悊椯旡掵拆巴忪彇弅弶徱庬搏帗恐庸廒懨彆";

wstring net_4_weight_pol = L"泣弡氵玑卆廤峵庝㸻旻娀㪉擞弛牾楆摛摼患懽浄彦悸䠂䰽戞䵺晫怒廜杖投䭦⑌䣺汛⟑栨嵧㉊澨攟堕樁ᬇ愉䅨䇗Ẑ䨯旅提橶怠惰恼澍彝爳抦廉悦搪ㆬ掇挣挪婓湻櫰呏擾侱柟枕氣悋從吳冄嫊夡咏氨䢝怔彷屍楘冿壇⋪徛怺拔呯䝷撧弔䪧扶僠渿悰牻乍掆您巋悘敝拈悬昜欛䪾嵩录弙瑒捥洖泈栘徶彾厢檴";

wstring net_4_bias_pol = L"柃嶅嵤媿";




struct Pos {
    int8_t x;
    int8_t y;
};

enum Dir {
    UP,
    DOWN,
    LEFT,
    RIGHT
};

struct Params0 {
    double death = -148.42012305961117;
    double size = 4.249118757407515;
    double dist = 17.495777101115365;
    double win = 142.95511897983494;
    double lose = -68.94264997675857;
    double flood = -21.891401061220208;
    double eat = 49.930044240674235;
    double lose_part = -6.9065559067686;
    double kill = 99.14748219536332;
    double kill_dude = -31.657405874965782;
    double cexplore = 0.5436079574741095;
};

struct Params1 {
    double death = -61.93605385655409;
    double size = 4.881137038784745;
    double dist = 12.48810009970582;
    double win = 192.94434651075196;
    double lose = -190.57311227738293;
    double flood = -49.99868548591225;
    double eat = 33.90698313151979;
    double lose_part = -19.984215693570313;
    double kill = 147.96318546156164;
    double kill_dude = -10.693103912807167;
    double cexplore = 0.5436079574741095; //0.6544837330252293;
};

struct Params {
    double death = -267.0341828063319;
    double size = 3.2065042230432246;
    double dist = 14.68825946000775;

    double win = 185.74016840752748;
    double lose = -62.4762792166449;
    double flood = -32.36377820117928;

    double eat = 45.70409988606824;
    double lose_part = -17.739604178471264;

    double kill = 73.46010334560454;
    double kill_dude = -12.231499714189257;

    double cexplore = 0.5436079574741095;
};

class Net {
public:
    int INPUT_SIZE = 24;
    int HIDDEN1 = 32;
    int HIDDEN2 = 16;
    int OUTPUT_SIZE = 1;

    std::vector<std::vector<float>> fc1_weight; // [32][24]
    std::vector<float> fc1_bias;                // [32]

    std::vector<std::vector<float>> fc2_weight; // [16][32]
    std::vector<float> fc2_bias;                // [16]

    std::vector<std::vector<float>> fc3_weight; // [1][16]
    std::vector<float> fc3_bias;                // [1]

    void init() {
        fc1_weight.resize(HIDDEN1, std::vector<float>(INPUT_SIZE, 0.0f));
        fc1_bias.resize(HIDDEN1, 0.0f);

        fc2_weight.resize(HIDDEN2, std::vector<float>(HIDDEN1, 0.0f));
        fc2_bias.resize(HIDDEN2, 0.0f);

        fc3_weight.resize(OUTPUT_SIZE, std::vector<float>(HIDDEN2, 0.0f));
        fc3_bias.resize(OUTPUT_SIZE, 0.0f);
    }

    float relu(float x) { return x > 0 ? x : 0; }

    std::vector<float> matvec(const std::vector<std::vector<float>>& W,
        const std::vector<float>& x,
        const std::vector<float>& b) {
        assert(W[0].size() == x.size());
        std::vector<float> y(W.size(), 0.0f);
        for (size_t i = 0; i < W.size(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < x.size(); ++j)
                sum += W[i][j] * x[j];
            y[i] = sum + b[i];
        }
        return y;
    }


    // calcule matvec normalement
    std::vector<float> matvecsoftmax(const std::vector<std::vector<float>>& W,
        const std::vector<float>& x,
        const std::vector<float>& b,
        bool apply_softmax = true) {
        assert(W[0].size() == x.size());
        std::vector<float> y(W.size(), 0.0f);

        // multiplication
        for (size_t i = 0; i < W.size(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < x.size(); ++j)
                sum += W[i][j] * x[j];
            y[i] = sum + b[i];
        }

        // appliquer softmax si demandé
        if (apply_softmax) {
            float max_val = *std::max_element(y.begin(), y.end());
            float sum_exp = 0.0f;
            for (auto& v : y) {
                v = std::exp(v - max_val); // stabilité numérique
                sum_exp += v;
            }
            for (auto& v : y) v /= sum_exp;
        }

        return y;
    }

    float forward(const std::vector<float>& input) {
        assert(input.size() == INPUT_SIZE);
        auto h1 = matvec(fc1_weight, input, fc1_bias);
        for (auto& v : h1) v = relu(v);
        auto h2 = matvec(fc2_weight, h1, fc2_bias);
        for (auto& v : h2) v = relu(v);
        auto out = matvec(fc3_weight, h2, fc3_bias);
        return out[0];
    }

    vector<float> forwardsoftmax(const std::vector<float>& input) {
        assert(input.size() == INPUT_SIZE);
        auto h1 = matvec(fc1_weight, input, fc1_bias);
        for (auto& v : h1) v = relu(v);
        auto h2 = matvec(fc2_weight, h1, fc2_bias);
        for (auto& v : h2) v = relu(v);
        auto out = matvecsoftmax(fc3_weight, h2, fc3_bias);
        return out;
    }

    // ----- fonction pour décoder wstring en poids -----
    static std::vector<float> decode_unicode_string_to_weightso(const std::wstring& ws,
        float offset = 24.0f,
        float divider = 1024.0f) {
        std::vector<uint8_t> bytes;
        for (wchar_t c : ws) {
            uint16_t val = static_cast<uint16_t>(c);
            bytes.push_back((val >> 8) & 0xFF); // octet haut
            bytes.push_back(val & 0xFF);        // octet bas
        }

        std::vector<float> weights;
        for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
            uint16_t s = (bytes[i] << 8) + bytes[i + 1];
            float val = (static_cast<float>(s) / divider) - offset;
            weights.push_back(val);
        }
        return weights;
    }


    static std::vector<float> decode_unicode_string_to_weights(const std::wstring& ws,
        float offset = 24.0f,
        float divider = 1024.0f)
    {

        int size = ws.size();

        string weights;
        for (auto& c : ws) {
            weights += (char)(c >> 8);
            weights += (char)(c & 255);
        }

        vector<float> output(size);
        stringstream ss(weights);
        for (int i = 0; i < size; i++) {
            char s1, s2;
            ss.get(s1);
            ss.get(s2);
            int32_t s = (uint8_t)s2 + (((uint8_t)s1) << 8);
            output[i] = (s / divider) - offset;
        }
        return output;



    }

    // ----- remplir le réseau -----
    void load_weights(const std::wstring& w_fc1_weight,
        const std::wstring& w_fc1_bias,
        const std::wstring& w_fc2_weight,
        const std::wstring& w_fc2_bias,
        const std::wstring& w_fc3_weight,
        const std::wstring& w_fc3_bias, float offset = 12.0, float divider = 2048.0)
    {
        auto vec = decode_unicode_string_to_weights(w_fc1_weight, offset, divider);
        for (int i = 0; i < HIDDEN1; ++i)
            for (int j = 0; j < INPUT_SIZE; ++j)
                fc1_weight[i][j] = vec[i * INPUT_SIZE + j];

        vec = decode_unicode_string_to_weights(w_fc1_bias, offset, divider);
        for (int i = 0; i < HIDDEN1; ++i) fc1_bias[i] = vec[i];

        vec = decode_unicode_string_to_weights(w_fc2_weight, offset, divider);
        for (int i = 0; i < HIDDEN2; ++i)
            for (int j = 0; j < HIDDEN1; ++j)
                fc2_weight[i][j] = vec[i * HIDDEN1 + j];

        vec = decode_unicode_string_to_weights(w_fc2_bias, offset, divider);
        for (int i = 0; i < HIDDEN2; ++i) fc2_bias[i] = vec[i];

        vec = decode_unicode_string_to_weights(w_fc3_weight, offset, divider);
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            for (int j = 0; j < HIDDEN2; ++j)
                fc3_weight[i][j] = vec[i * HIDDEN2 + j];

        vec = decode_unicode_string_to_weights(w_fc3_bias, offset, divider);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            fc3_bias[i] = vec[i];
            cerr << "bias=" << vec[i] << endl;
        }

    }
};



const int OPP[4] = { 1,0,3,2 };

const string direction[4] = { "UP", "DOWN", "LEFT", "RIGHT" };

const int DX[4] = { 0,0,-1,1 };
const int DY[4] = { -1,1,0,0 };

const int MAX_BODY = 256;

struct Snake {

    int id;
    bool alive;

    int dir;

    int head;
    int tail;
    int len;

    Pos body[MAX_BODY];

};

inline Pos headPos(const Snake& s) {
    return s.body[s.head];
}

inline Pos tailPos(const Snake& s) {
    return s.body[s.tail];
}


const int MAX_W = 45;
const int MAX_H = 30;
const int BORDER = 0;
const int BORDERH = 5;

enum Cell {
    EMPTY = 0,
    WALL = 1 << 0,
    SNAKE = 1 << 1,
    ENERGY = 1 << 2,
    GENERGY = 1 << 3
};

struct Grid {

    uint8_t cell[MAX_H + BORDERH][MAX_W + BORDER * 2];

};

const int MAX_SNAKES = 8;
const int MAX_POWER = 400;

struct GameState {

    int w, h;

    Grid gridu[8];

    Grid grid;

    int snakeCount;
    Snake snakes[8];

    int energyCount;
    Pos energy[MAX_POWER];

};

struct Node {
    int parent;

    int first_child;
    int child_count;

    double score = 0.0;
    int visits = 0;
    int move;
    double mult = 1.0;
    double prior = 0.0;

};

struct NodeB : Node {
    GameState game;
    Pos wallenergy = { -1, -1 };
};

// encode_state_full en C++
std::vector<float> encode_state_full(
    const Snake& snake,
    const std::vector<Pos>& energies,
    const std::vector<Snake>& enemies,
    int W, int H, int MAX_LEN,
    int max_energies = 5,
    int max_enemies = 3,
    const Pos* prev_dir = nullptr,
    float flood = 0.0f
) {
    std::vector<float> state_vec;

    Pos h = snake.body[snake.head];
    int hx = h.x;
    int hy = h.y;

    // --- position de la tête
    state_vec.push_back(static_cast<float>(hx) / W);
    state_vec.push_back(static_cast<float>(hy) / H);

    // --- encode distances aux énergies
    std::vector<Pos> energies_sorted = energies;
    std::sort(energies_sorted.begin(), energies_sorted.end(), [hx, hy](const Pos& a, const Pos& b) {
        return std::abs(a.x - hx) + std::abs(a.y - hy) < std::abs(b.x - hx) + std::abs(b.y - hy);
        });

    for (int i = 0; i < max_energies; ++i) {
        if (i < energies_sorted.size()) {
            int ex = energies_sorted[i].x;
            int ey = energies_sorted[i].y;
            state_vec.push_back(static_cast<float>(ex - hx) / W);
            state_vec.push_back(static_cast<float>(ey - hy) / H);
        }
        else {
            state_vec.push_back(0.0f);
            state_vec.push_back(0.0f);
        }
    }

    // --- encode distances aux ennemis
    std::vector<Snake> enemies_sorted = enemies;
    std::sort(enemies_sorted.begin(), enemies_sorted.end(), [hx, hy](const Snake& a, const Snake& b) {
        Pos ha = a.body[a.head];
        Pos hb = b.body[b.head];
        return std::abs(ha.x - hx) + std::abs(ha.y - hy) < std::abs(hb.x - hx) + std::abs(hb.y - hy);
        });

    for (int i = 0; i < max_enemies; ++i) {
        if (i < enemies_sorted.size()) {
            Pos eh = enemies_sorted[i].body[enemies_sorted[i].head];
            float len_ratio = static_cast<float>(enemies_sorted[i].len) / MAX_LEN;
            state_vec.push_back(static_cast<float>(eh.x - hx) / W);
            state_vec.push_back(static_cast<float>(eh.y - hy) / H);
            state_vec.push_back(len_ratio);
        }
        else {
            state_vec.push_back(0.0f);
            state_vec.push_back(0.0f);
            state_vec.push_back(0.0f);
        }
    }

    // --- taille du serpent
    state_vec.push_back(static_cast<float>(snake.len) / MAX_LEN);

    // --- flood fill (optionnel)
    // state_vec.push_back(flood / 30.0f);

    // --- orientation précédente
    if (prev_dir == nullptr) {
        state_vec.push_back(0.0f);
        state_vec.push_back(0.0f);
    }
    else {
        state_vec.push_back(static_cast<float>(prev_dir->x));
        state_vec.push_back(static_cast<float>(prev_dir->y));
    }

    return state_vec;
}



const int MAX_NODE = 50000;
const int MAX_CHILD = 50000;

struct SM {

    Node nodes[MAX_NODE];
    NodeB nodesb[10];
    int children[MAX_CHILD];

    int ldir[8];

    int nodeCount = 0;
    int childCount = 0;

    int ITER = 0;

    Params params;

    Net net;
    Net netpuct;

    void loadWeights() {
        net.init();
        net.load_weights(model_0_weight, model_0_bias, model_2_weight, model_2_bias, model_4_weight, model_4_bias);


    }

    void loadWeightsPUCT() {

        netpuct.INPUT_SIZE = 24;
        netpuct.HIDDEN1 = 64;
        netpuct.HIDDEN2 = 32;
        netpuct.OUTPUT_SIZE = 4;
        netpuct.init();
        netpuct.load_weights(net_0_weight_pol, net_0_bias_pol, net_2_weight_pol, net_2_bias_pol, net_4_weight_pol, net_4_bias_pol);


    }

    int createNode(int parent = -1) {

        int id = nodeCount;
        nodeCount = nodeCount + 1;

        nodes[id].parent = parent;
        nodes[id].first_child = -1;
        nodes[id].child_count = 0;
        nodes[id].score = 0.0;
        nodes[id].visits = 0;
        nodes[id].move = UP;
        nodes[id].mult = 1.0;

        return id;
    }


    void addChild(int parent, int child) {

        if (nodes[parent].first_child == -1)
            nodes[parent].first_child = childCount;

        children[childCount++] = child;
        nodes[parent].child_count++;
    }



    int selection2(int node) {

        int start = nodes[node].first_child;
        int count = nodes[node].child_count;

        if (count == 0) return node;   // sécurité

        long double best_ucb = -1e18;
        int best_child = children[start];  // initialiser correctement

        long double log_parent = log(nodes[node].visits + 1);

        for (int i = 0; i < count; i++) {

            int c = children[start + i];

            long double ucb;

            if (nodes[c].visits == 0) {
                ucb = 1e18;
            }
            else {

                /*
                long double c_explore = 0.1; // facteur d’exploration
                */

                long double c_explore = params.cexplore; // facteur d’exploration
                long double exploit = nodes[c].score / (long double)(nodes[c].visits);
                long double explore = c_explore * sqrt(log_parent / (long double)nodes[c].visits);
                ucb = exploit + explore;
                ucb *= nodes[c].mult;
            }

            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = c;
            }
        }

        return best_child;
    }

    double adaptCExplore(int nb_player, double baseC)
    {
        // baseC = params.cexplore, par ex 0.55

        // si 1 joueur → on reste proche de baseC
        // si 2-4 joueurs → on augmente légèrement
        double factor = 1.0 + 0.1 * (nb_player - 1); // simple multiplicateur

        double c = baseC * factor;

        // clamp entre 0.1 et 1.0 pour rester raisonnable
        if (c > 1.0) c = 1.0;
        if (c < 0.1) c = 0.1;

        return c;
    }

    int selection(int node, double nb_player = 0.0) {

        int start = nodes[node].first_child;
        int count = nodes[node].child_count;

        if (count == 0) return node;

        /*if(rand() % 100 <= 2){
            return children[start + rand() % count];
        }*/

        Node* n = nodes;

        double best_ucb = -1e30;
        int best_child = children[start];

        double log_parent = log((double)(n[node].visits + 1));
        double c_explore = params.cexplore;//adaptCExplore(nb_player, params.cexplore);

        for (int i = 0; i < count; i++) {

            int c = children[start + i];
            Node& nc = n[c];

            double inv_visits = (nc.visits > 0) ? (1.0 / nc.visits) : 0.0;

            double exploit = nc.score * inv_visits;
            double explore = c_explore * sqrt(log_parent * inv_visits);

            double ucb = (nc.visits == 0 ? 1e30 : (exploit + explore) * nc.mult);

            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = c;
            }
        }

        return best_child;
    }

    
    void expand(int node, GameState& game, int ind_snake, Pos energy[MAX_POWER], int depth, int opp_snake_len = 0) {

        Grid &grcurp = game.grid;

        std::vector<int> moves;

        for (int i = 0; i < 4; i++) {

            Pos h = headPos(game.snakes[ind_snake]);

            int nx = h.x + DX[i];
            int ny = h.y + DY[i];
            if (nx < 0 || ny < 0 || nx >= game.w || ny >= game.h)
                continue;

            if (grcurp.cell[ny][nx] & WALL)continue;


            if (i == UP && grcurp.cell[ny][nx] == EMPTY && isVertical(game.snakes[ind_snake]))
                continue;

             
            int curid = (game.snakes[ind_snake].head - 1 + MAX_BODY) % MAX_BODY;
            Pos cur = game.snakes[ind_snake].body[curid];
            if (nx == cur.x && ny == cur.y) {
                continue;
            }

            moves.push_back(i);

            /*
            int child = createNode(node);
            nodes[child].move = i;
            if (grcurp.cell[ny][nx] & ENERGY)
                nodes[child].mult = params.eat;
            addChild(node, child);
            */



        }

        std::vector<std::pair<double,int>> scored;

        for(int m : moves) {
            double ps = policyScore(game, ind_snake, m, energy);
            scored.emplace_back(ps, m);
        }

        if(moves.empty())return;

        std::sort(scored.begin(), scored.end(),
            [](auto &a, auto &b){
                return a.first > b.first;
            });

        int TOPK = 2;
        if(!scored.empty() && scored[0].first > scored[1].first + 1000.0)
            TOPK--; // greedy direct

        //if(depth > 3)TOPK--;

        for(int k = 0;k < TOPK && k < scored.size();++k){
            int m = scored[k].second;

            Pos h = game.snakes[ind_snake].body[game.snakes[ind_snake].head];

            int nx = h.x + DX[m];
            int ny = h.y + DY[m];    

            int child = createNode(node);
            nodes[child].move = m;
            if (grcurp.cell[ny][nx] & ENERGY)
                nodes[child].mult = params.eat;
            addChild(node, child);

        }


    }

    int selection_puct(int node_idx) {
        Node& n = nodes[node_idx];

        int start = n.first_child;
        int count = n.child_count;

        if (count == 0)
            return node_idx; // sécurité si pas d'enfant

        double best_score = -1e18;
        int best_child_idx = children[start];

        double log_parent = std::log(n.visits + 1.0);

        // facteur d'exploration très petit (PUCT soft)
        double soft_factor = 1.0; // ~5% influence de l'exploration

        for (int i = 0; i < count; ++i) {
            int c_idx = children[start + i];
            Node& child = nodes[c_idx];

            // Q-value (exploitation)
            double Q = 0.0;
            if (child.visits > 0)
                Q = child.score / child.visits;

            // Prior / exploration U
            double P = child.prior; // tu dois remplir child.prior avant l'appel
            double U = soft_factor * params.cexplore * P * std::sqrt(log_parent) / (1.0 + child.visits);

            double ucb = Q + U;
            ucb *= child.mult; // multiplicateur spécifique à ton système

            if (ucb > best_score) {
                best_score = ucb;
                best_child_idx = c_idx;
            }
        }

        return best_child_idx;
    }

    int selection_puctf(int node_idx) {
        Node& n = nodes[node_idx];

        int start = n.first_child;
        int count = n.child_count;

        if (count == 0)
            return node_idx; // sécurité si pas d'enfant

        double best_score = -1e18;
        int best_child_idx = children[start];

        double log_parent = std::log(n.visits + 1.0);

        for (int i = 0; i < count; ++i) {
            int c_idx = children[start + i];
            Node& child = nodes[c_idx];

            double ucb;

            if (child.visits == 0) {
                // forcer les enfants non visités à être explorés
                ucb = 1e18;
            }
            else {
                // Q-value (exploitation)
                double Q = child.score / child.visits;

                // Prior / exploration U
                double P = child.prior; // doit être rempli avant
                double U = params.cexplore * P * std::sqrt(log_parent) / (1.0 + child.visits);

                ucb = (Q + U) * child.mult;
            }

            if (ucb > best_score) {
                best_score = ucb;
                best_child_idx = c_idx;
            }
        }

        return best_child_idx;
    }

    void expandPUCT(int node_idx, const GameState& game, int ind_snake,
        const std::vector<float>* policy_priors = nullptr)
    {
        Grid grcurp = game.grid;
        Snake snake = game.snakes[ind_snake];
        Pos h = snake.body[snake.head]; // tête

        // --- moves normaux ---
        for (int i = 0; i < 4; ++i) {
            int nx = h.x + DX[i];
            int ny = h.y + DY[i];

            if (nx < 0 || ny < 0 || nx >= game.w || ny >= game.h)
                continue;

            if (grcurp.cell[ny][nx] & WALL)
                continue;

            /*int sf = get_safe_moves(nx, ny, game.grid, game.w, game.h);
            if (sf == 0)
                continue;*/

            if (i == 0 && grcurp.cell[ny][nx] == EMPTY && isVertical(snake))
                continue;

            // collision corps
            /*bool coll = false;
            for (int k = 0; k < snake.len; ++k) {
                int curid = (snake.tail + k) % MAX_BODY;
                Pos cur = snake.body[curid];
                if (nx == cur.x && ny == cur.y) {
                    coll = true;
                    break;
                }
            }
            if (coll) continue;*/

            int curid = (game.snakes[ind_snake].head - 1 + MAX_BODY) % MAX_BODY;
            Pos cur = game.snakes[ind_snake].body[curid];
            if (nx == cur.x && ny == cur.y) {
                continue;
            }


            int child = createNode(node_idx);
            nodes[child].move = i;

            if (grcurp.cell[ny][nx] & ENERGY)
                nodes[child].mult = params.eat;

            // prior
            if (policy_priors)
                nodes[child].prior = (*policy_priors)[i];
            else
                nodes[child].prior = 0.25f; // uniform

            addChild(node_idx, child);
        }

        // --- fallback (aucun move valide) ---
        /*if (nodes[node_idx].child_count == 0) {
            int bestDir = 0;
            int minDist = 1e9;
            Pos tail = snake.body[snake.tail];

            for (int i = 0; i < 4; ++i) {
                int nx = h.x + DX[i];
                int ny = h.y + DY[i];

                if (nx < 0 || ny < 0 || nx >= game.w || ny >= game.h)
                    continue;
                if (grcurp.cell[ny][nx] & WALL)
                    continue;
                if (isVertical(snake) && i == 0 && grcurp.cell[ny][nx] == EMPTY)
                    continue;

                bool coll = false;
                for (int k = 1; k < snake.len; ++k) { // skip head
                    int curid = (snake.tail + k) % MAX_BODY;
                    Pos cur = snake.body[curid];
                    if (nx == cur.x && ny == cur.y) {
                        coll = true;
                        break;
                    }
                }
                if (coll) continue;

                int dist = std::abs(nx - tail.x) + std::abs(ny - tail.y);
                if (dist < minDist) {
                    minDist = dist;
                    bestDir = i;
                }
            }

            int child = createNode(node_idx);
            nodes[child].move = bestDir;
            nodes[child].prior = policy_priors ? (*policy_priors)[bestDir] : 0.25f;
            addChild(node_idx, child);
        }*/
    }

    std::vector<float> get_policy_priors2(Net& model, int snake_idx, const GameState& game, int opp_len) {
        // --- Récupérer le snake principal ---
        Snake snake = game.snakes[snake_idx];

        // --- Récupérer l'autre snake ---
        vector<Snake> opp_snake;
        if (snake_idx < game.snakeCount) {
            for (int i = 0; i < game.snakeCount; ++i) {
                if (game.snakes[i].alive) {
                    opp_snake.push_back(game.snakes[i]);
                    break;

                }
            }

        }
        else {
            for (int i = game.snakeCount; i < game.snakeCount + opp_len; ++i) {
                if (game.snakes[i].alive) {
                    opp_snake.push_back(game.snakes[i]);
                    break;

                }
            }

        }

        // --- Positions de l'énergie ---
        vector<Pos> energy;
        for (int ind = 0; ind < game.energyCount; ++ind) {
            Pos e = game.energy[ind];
            if (game.grid.cell[e.y][e.x] & ENERGY)
                energy.push_back(game.energy[ind]);

        }

        // --- Encoder l'état en vecteur float ---
        std::vector<float> state = encode_state_full(game.snakes[snake_idx], energy, opp_snake, game.w, game.h, 256);

        // --- Forward softmax pour obtenir les logits comme probs ---
        std::vector<float> logits = model.forwardsoftmax(state); // size=4

        // --- Masquer les moves invalides ---
        const Pos& head = snake.body[snake.head];
        std::vector<bool> mask(4, false);
        for (int i = 0; i < 4; ++i) {
            int nx = head.x + DX[i];
            int ny = head.y + DY[i];

            if (nx < 0 || ny < 0 || nx >= game.w || ny >= game.h)
                mask[i] = true;
            else if (game.grid.cell[ny][nx] & WALL)
                mask[i] = true;
            else if (get_safe_moves(nx, ny, game.grid, game.w, game.h) == 0)
                mask[i] = true;
            else {
                bool coll = false;
                for (int k = 1; k < snake.len; ++k) { // skip head
                    int curid = (snake.tail + k) % MAX_BODY;
                    Pos cur = snake.body[curid];
                    if (nx == cur.x && ny == cur.y) {
                        coll = true;
                        break;
                    }
                }
                if (coll) mask[i] = true;
            }
        }

        // --- Appliquer le masque et renormaliser ---
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            if (mask[i]) logits[i] = 0.0f;
            sum += logits[i];
        }

        if (sum > 0) {
            for (auto& v : logits) v /= sum;
        }
        else {
            // fallback: moves valides uniformes
            int valid = 0;
            for (bool m : mask) if (!m) valid++;
            for (int i = 0; i < 4; ++i)
                if (!mask[i]) logits[i] = 1.0f / valid;
                else logits[i] = 0.0f;
        }

        return logits; // vector<float> taille 4
    }

    std::vector<float> get_policy_priors(Net& model, int snake_idx, const GameState& game, int opp_len) {
        // --- Récupérer le snake principal ---
        Snake snake = game.snakes[snake_idx];

        // --- Récupérer les snakes adverses ---
        vector<Snake> opp_snake;
        if (snake_idx < game.snakeCount) {
            for (int i = 0; i < game.snakeCount; ++i) {
                if (game.snakes[i].alive) {
                    opp_snake.push_back(game.snakes[i]);
                    break;

                }
            }

        }
        else {
            for (int i = game.snakeCount; i < game.snakeCount + opp_len; ++i) {
                if (game.snakes[i].alive) {
                    opp_snake.push_back(game.snakes[i]);
                    break;

                }
            }

        }

        // --- Positions de l'énergie ---
        std::vector<Pos> energy;
        for (int ind = 0; ind < game.energyCount; ++ind) {
            Pos e = game.energy[ind];
            if (game.grid.cell[e.y][e.x] & ENERGY)
                energy.push_back(e);
        }

        // --- Encoder l'état en vecteur float ---
        std::vector<float> state = encode_state_full(snake, energy, opp_snake, game.w, game.h, 256);

        // --- Forward softmax pour obtenir directement les probabilités ---
        std::vector<float> policy_priors = model.forwardsoftmax(state); // size = 4

        return policy_priors; // array de 4 floats sommant à 1
    }

    inline int move_cost(int dir) {
        //if(dir <= 1)return 2;
        return 1; // ou autre si tu veux pondérer
    }

    vector<vector<vector<int>>> compute_distance_mapgo(
        int W, int H,
        const Grid& grid,
        const Pos energy[400],
        int max_energy
    ) {
        vector<vector<vector<int>>> dm(W * H); // FIX

        for (int ie = 0; ie < max_energy; ++ie) {

            int sx = energy[ie].x;
            int sy = energy[ie].y;

            vector<vector<int>> dist(H, vector<int>(W, 1e9));

            priority_queue<
                tuple<int, int, int>,
                vector<tuple<int, int, int>>,
                greater<>
            > pq;

            dist[sy][sx] = 0;
            pq.emplace(0, sx, sy);

            while (!pq.empty()) {
                auto [d, x, y] = pq.top();
                pq.pop();

                if (d != dist[y][x]) continue;

                for (int i = 0; i < 4; ++i) {
                    int nx = x + DX[i];
                    int ny = y + DY[i];

                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;

                    if (grid.cell[ny][nx] == WALL)
                        continue;

                    if (grid.cell[ny][nx] == ENERGY)
                        continue;

                    int nd = d + move_cost(i);

                    if (nd < dist[ny][nx]) {
                        dist[ny][nx] = nd;
                        pq.emplace(nd, nx, ny);
                    }
                }
            }

            dm[sy * W + sx] = std::move(dist); // important (perf)
        }

        return dm;
    }

    vector<vector<vector<int>>> compute_distance_mapg(
        int W, int H,
        const Grid& grid,
        const Pos energy[400],
        int max_energy,
        vector<vector<vector<pair<int, int>>>>& parentg // 🔥 ajouté
    ) {
        vector<vector<vector<int>>> dm(W * H);
        parentg.clear();
        parentg.resize(W * H); // 🔥 allocation

        for (int ie = 0; ie < max_energy; ++ie) {

            int sx = energy[ie].x;
            int sy = energy[ie].y;

            vector<vector<int>> dist(H, vector<int>(W, 1e9));
            vector<vector<pair<int, int>>> parent(H, vector<pair<int, int>>(W, { -1,-1 }));

            priority_queue<
                tuple<int, int, int>,
                vector<tuple<int, int, int>>,
                greater<>
            > pq;

            dist[sy][sx] = 0;
            pq.emplace(0, sx, sy);

            while (!pq.empty()) {
                auto [d, x, y] = pq.top();
                pq.pop();

                if (d != dist[y][x]) continue;

                for (int i = 0; i < 4; ++i) {
                    int nx = x + DX[i];
                    int ny = y + DY[i];

                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;

                    if (grid.cell[ny][nx] == WALL)
                        continue;

                    if (grid.cell[ny][nx] == ENERGY)
                        continue;

                    int nd = d + move_cost(i);

                    if (nd < dist[ny][nx]) {
                        dist[ny][nx] = nd;

                        // 🔥 on stocke le parent
                        parent[ny][nx] = { x, y };

                        pq.emplace(nd, nx, ny);
                    }
                }
            }

            int id = sy * W + sx;
            dm[id] = std::move(dist);
            parentg[id] = std::move(parent); // 🔥 stockage parent
        }

        return dm;
    }

    vector<Pos> reconstruct_path(
        int tx, int ty,
        const vector<vector<pair<int, int>>>& parent
    ) {
        vector<Pos> path;

        int x = tx, y = ty;

        while (x != -1 && y != -1) {
            path.push_back({ x, y });
            auto [px, py] = parent[y][x];
            x = px;
            y = py;
        }

        reverse(path.begin(), path.end());
        return path;
    }

    double policyScore(const GameState& game, int player, int m, Pos energy[MAX_POWER]) {
        const Snake& s = game.snakes[player];
        Pos h = s.body[s.head];

        int nx = h.x + DX[m];
        int ny = h.y + DY[m];

        double score = 0.0;

        // --- aller vers énergie
        for(int i = 0; i < game.energyCount; i++) {
            Pos e = energy[i];
            if(game.grid.cell[e.y][e.x] & ENERGY) {
                //int d = abs(nx - e.x) + abs(ny - e.y);
                //score += 10.0 / (d + 1);
                score += 10.0 / (distg[e.y*game.w+e.x][ny][nx] + 1);
                
                if (nx == e.x && ny == e.y)score += 1000.0;
            }
        }

        // --- éviter murs
        if(nx <= 1 || ny <= 1 || nx >= game.w-2 || ny >= game.h-2)
            score -= 5.0;

        // --- éviter cul-de-sac
        int free = get_safe_moves(nx, ny, game.grid, game.w, game.h);
        score += free * 2.0;

        // --- stabilité direction
        //if(m == s.dir)
        //    score += 1.5;

        return score;
    }


    void playMoveTurn(GameState& g, double score[8], int opp_len) {

        int total = g.snakeCount + opp_len;

        vector<pair<int, int>> ven;

        for (int id = 0; id < total; ++id) {


            Snake& s = g.snakes[id];

            if (!s.alive)continue;

            Pos h = s.body[s.head];
            int move = s.dir;
            int nx = h.x + DX[move];
            int ny = h.y + DY[move];

            // --- MOVE NORMAL
            s.tail = (s.tail + 1) % MAX_BODY;
            s.head = (s.head + 1) % MAX_BODY;
            s.body[s.head] = { (int8_t)nx,(int8_t)ny };

            // --- ENERGY
            if (g.grid.cell[ny][nx] & ENERGY) {
                s.tail = (s.tail - 1 + MAX_BODY) % MAX_BODY;
                ven.push_back({ nx, ny });
                s.len++;
                score[id] += params.eat; //10
            }
            /*else if (g.grid.cell[ny][nx] & WALL) {
                s.head = (s.head - 2 + MAX_BODY) % MAX_BODY;
                s.len--;
                if(s.len < 3)s.alive = false;
                score[id] += params.lose_part; //10
            }*/



        }

        for (int i = 0; i < ven.size(); ++i) {
            int nx = ven[i].first, ny = ven[i].second;
            g.grid.cell[ny][nx] &= ~ENERGY;
        }

        for (int id = 0; id < total; ++id) {
            // --- CHECK COLLISION AVANT MOVE
            bool collision = false;

            Snake& s = g.snakes[id];
            if (!s.alive)continue;


            Pos h = s.body[s.head];
            int nx = h.x, ny = h.y;
            int ind_other = -1;
            bool head = false;
            for (int j = 0; j < total; ++j) {
                int end = 0;
                if (id == j)end = 1;
                Snake& other = g.snakes[j];
                if (!other.alive)continue;

                int cur = other.tail;
                for (int k = 0; k < other.len - end; ++k) {
                    Pos b = other.body[cur];
                    if (b.x == nx && b.y == ny) {
                        ind_other = j;
                        if (k == other.len - 1)head = true;
                        collision = true;
                        break;
                    }
                    cur = (cur + 1) % MAX_BODY;
                }
                if (collision) break;
            }

            if (collision) {
                // 👉 rollback : la tête devient segment précédent
                int prev = (s.head - 1 + MAX_BODY) % MAX_BODY;
                s.head = prev;
                s.len--; // shrink
                if (s.len < 3)
                    s.alive = false;
                else
                    score[id] += params.lose_part;
                if (head) {
                    Snake& so = g.snakes[ind_other];
                    int prev = (so.head - 1 + MAX_BODY) % MAX_BODY;
                    so.head = prev;
                    so.len--; // shrink
                    if (so.len < 3) {
                        so.alive = false;
                        if (s.alive && id < g.snakeCount && ind_other >= g.snakeCount)score[id] += params.kill;
                        if (s.alive && id < g.snakeCount && ind_other < g.snakeCount)score[id] += params.kill_dude;
                        if (s.alive && id >= g.snakeCount && ind_other >= g.snakeCount)score[id] += params.kill_dude;
                        if (s.alive && id >= g.snakeCount && ind_other < g.snakeCount)score[id] += params.kill;
                    }
                    else if (s.len > so.len) {
                        score[id] += params.lose_part * -2.0;
                    }


                }


            }


        }



        /*while(true){

            Grid grcurp = g.grid;

            for(int is =  0;is < g.snakeCount+opp_len;++is){
                for(int ids = 0; ids < g.snakes[is].len; ids++){
                    int id = (g.snakes[is].tail + ids) % MAX_BODY;
                    Pos p = g.snakes[is].body[id];

                    if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                        grcurp.cell[p.y][p.x] |= SNAKE;
                }

            }

            bool stop = true;

            for(int i  = 0;i < total;++i){
                int fall = applyGravity(g, g.snakes[i], grcurp);
                if(fall != INT_MAX){
                    stop = false;
                }
            }

            if(stop)break;

        }*/

        doFalls2(g, total);
        /*
        bool needIntercoil = false;

        for(int i=0;i<total;i++){
            for(int j=i+1;j<total;j++){
                if(touchingVertical2(g.snakes[i], g.snakes[j])){
                    needIntercoil = true;
                    break;
                }
            }
            if(needIntercoil) break;
        }

        if(needIntercoil){
            doIntercoiledFalls(g, total);
        }*/


    }

    std::vector<int> expandBeamP(const Grid &grcurp, Snake &snake, int wt, int ht) {

        std::vector<int> moves;

        // head
        Pos h = snake.body[snake.head];

        for (int i = 0; i < 4; i++) {

            int nx = h.x + DX[i];
            int ny = h.y + DY[i];

            // hors map
            if (nx < 0 || ny < 0 || nx >= wt || ny >= ht)
                continue;

            // mur
            if (grcurp.cell[ny][nx] & WALL)
                continue;

            // règle verticale
            if (i == 0 && grcurp.cell[ny][nx] == EMPTY && isVertical(snake))
                continue;

            // éviter revenir sur le cou
            int curid = (snake.head - 1 + MAX_BODY) % MAX_BODY;
            const Pos &cur = snake.body[curid];
            if (nx == cur.x && ny == cur.y)
                continue;

            // priorité énergie
            if (grcurp.cell[ny][nx] & ENERGY) {
                return {i};  // retour direct
            }

            moves.push_back(i);
        }

        return moves;
    }

    int getDirection(const Snake& s) {

        Pos head = s.body[s.head];
        int next = (s.head - 1 + MAX_BODY) % MAX_BODY;
        Pos neck = s.body[next];

        int dx = head.x - neck.x;
        int dy = head.y - neck.y;

        if (dx == 0 && dy == -1) return 0; // UP
        if (dx == 0 && dy == 1) return 1; // DOWN
        if (dx == -1 && dy == 0) return 2; // LEFT
        if (dx == 1 && dy == 0) return 3; // RIGHT

        return -1;
    }

    vector<vector<vector<int>>> distg;
    vector<vector<vector<pair<int, int>>>> parentg_sim;


    string Play(int height, int width, Grid grid, Snake my_snake[8], int my_snake_len, Snake opp_snake[8], int opp_snake_len, int my_id_snake[8], Pos energy[MAX_POWER], int power_source_count, int time) {

        auto startm = high_resolution_clock::now();;
        int maxt = 0;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = duration.count();
            return(maxt <= time);
        };


        int totalSnake = my_snake_len + opp_snake_len;

        nodeCount = 0;
        childCount = 0;

        int root[8];
        for (int i = 0; i < totalSnake; ++i) {
            root[i] = createNode(-1);
        }

        int DEPTH = 9 - my_snake_len;
        int turn = 0;
        int make_rand = 0;

        cerr << "TSRTA" << endl;

        while (getTime()) {
            //cerr << maxt << endl;

            int node[8];
            for (int i = 0; i < totalSnake; ++i) {
                node[i] = root[i];
            }

            GameState game;
            game.w = width + BORDER * 2;
            game.h = height + BORDERH;
            game.snakeCount = my_snake_len;
            for (int i = 0; i < my_snake_len; ++i) {
                game.snakes[i] = my_snake[i];
                game.snakes[i].dir = UP;
                game.snakes[i].alive = true;

            }
            int inds = 0;
            for (int i = my_snake_len; i < my_snake_len + opp_snake_len; ++i) {
                game.snakes[i] = opp_snake[inds];
                game.snakes[i].dir = UP;
                game.snakes[i].alive = true;
                inds++;
            }

            /*for (int i = 0; i < power_source_count; ++i) {
                game.energy[i] = energy[i];

            }*/
            game.grid = grid;

            game.energyCount = power_source_count;

            double score[8] = { 0,0,0,0,0,0,0,0 };

            //DEPTH = (turn < 1000) ? (3 + turn / 250) : (7 + (turn - 1000) / 1000);

            for (int depth = 0; depth < DEPTH; ++depth) {
                //cerr << "depth=" << depth << endl;
                for (int i = 0; i < totalSnake; ++i) {

                    if (!game.snakes[i].alive)continue;

                    if(i < game.snakeCount+opp_snake_len){

                        if (nodes[node[i]].first_child == -1) {
                            Pos h = headPos(game.snakes[i]);
                            expand(node[i], game, i, energy, depth, opp_snake_len);
                            //vector<float> priors = get_policy_priors(netpuct, i, game, opp_snake_len);
                            //expandPUCT(node[i], game, i, &priors);

                            if (nodes[node[i]].child_count == 0) {
                                game.snakes[i].dir = getDirection(game.snakes[i]);
                                continue;
                            }

                        }

                        node[i] = selection(node[i]);
                        //node[i] = selection_puct(node[i]);

                        game.snakes[i].dir = nodes[node[i]].move;
                        //cerr << "mv=" << nodes[node[i]].move << endl;
                    }
                    else{
                        
                        int  iplayer = i;

                        std::vector<int> moveo = expandBeamP(game.grid, game.snakes[iplayer], game.w, game.h);

                        if (!moveo.empty()){
                            std::vector<std::pair<double,int>> scored;

                            for(int m : moveo) {
                                double ps = policyScore(game, iplayer, m, energy);
                                scored.emplace_back(ps, m);
                            }

                            std::sort(scored.begin(), scored.end(),
                                [](auto &a, auto &b){
                                    return a.first > b.first;
                                });

                                
                            game.snakes[iplayer].dir = scored[0].second;

                        }
                        else{
                            game.snakes[iplayer].dir = getDirection(game.snakes[iplayer]);
                        }
                        

                    }


                }

                playMoveTurn(game, score, opp_snake_len);

                int count = 0;
                int bestDist = 1e9;
                for (int ie = 0;ie < game.energyCount;ie++) {
                    Pos e = energy[ie];
                    if (game.grid.cell[e.y][e.x] == ENERGY) {
                        count++;

                    }
                }
                if (count == 0)break;


            }

            //rolloutSMITS(game, score, opp_snake_len, 1);

            //cerr << "evaluation" << endl;

            int tot_team = 0, tot_opp = 0;
            for (int i = 0; i < totalSnake; ++i) {
                Snake& s = game.snakes[i];
                if (s.alive) {
                    if (i < game.snakeCount)tot_team++;
                    else tot_opp++;
                }
            }

            Grid grcurp = game.grid;

            for (int is = 0; is < game.snakeCount + opp_snake_len; ++is) {
                if (!game.snakes[is].alive)continue;
                for (int ids = 0; ids < game.snakes[is].len; ids++) {
                    int id = (game.snakes[is].tail + ids) % MAX_BODY;
                    Pos p = game.snakes[is].body[id];

                    if (p.y >= 0 && p.y < game.h && p.x >= 0 && p.x < game.w)
                        grcurp.cell[p.y][p.x] |= SNAKE;
                }

            }

            /*vector<Pos> energy;
            for (int ind = 0; ind < power_source_count; ++ind) {
                Pos e = energy[ind];
                if (game.grid.cell[e.y][e.x] & ENERGY)
                    energy.push_back(e);

            }*/


            for (int i = 0; i < totalSnake; ++i) {
                Snake& s = game.snakes[i];
                int hx = s.body[s.head].x;
                int hy = s.body[s.head].y;

                double sc = score[i];

                //double sign = 1;
                //if(i >= game.snakeCount)sign = -1;

                // 1️⃣ Mort = penalty mais pas trop violent
                if (!s.alive) {
                    sc = params.death;//-200;           // pas -1 direct
                    backprop(node[i], (sc / 200.0));
                    continue;
                }





                // 2️⃣ Taille du serpent
                sc += s.len * params.size;

                // 3️⃣ Distance à la nourriture la plus proche
                int count = 0;
                int bestDist = 1e9;
                double sumScore = 0.0;
                for (int ind = 0; ind < power_source_count; ++ind) {
                    Pos e = energy[ind];
                    if ((game.grid.cell[e.y][e.x] == ENERGY) && hy >= 0 and hy < game.h && hx >= 0 && hx < game.w) {
                        int w = (width + BORDER * 2);
                        int distp = 0;
                        int gap = compute_max_gap(hx, hy, parentg_sim[e.y * w + e.x], grcurp, game.w, game.h, distp);

                        if (gap > s.len - 2)gap += 1000;

                        int d = distp + gap;//abs(e.x - hx) + abs(e.y - hy);distg[e.y*w+e.x][hy][hx]
                        //cerr << "d=" << d << endl;
                        count++;
                        //int db = distg[e.y * w + e.x][hy][hx];
                        if (d < bestDist) bestDist = d;
                        sumScore += params.dist / (d + 1);//10.0

                    }

                }

                if (bestDist < 1e9) {
                    sc += params.dist * 30.0 / (bestDist + 1);//10.0
                    //sc += 1.0 * sumScore;

                }

                sc += 0.3 * sumScore;

                if (count == 0) {
                    if (i < game.snakeCount && tot_team > tot_opp)sc += params.win;
                    if (i < game.snakeCount && tot_team < tot_opp)sc += params.lose;
                    if (i >= game.snakeCount && tot_team < tot_opp)sc += params.win;
                    if (i >= game.snakeCount && tot_team > tot_opp)sc += params.lose;

                }


                /*
                vector<Snake> enemies;
                if(i < game.snakeCount){
                    for(int j = game.snakeCount;j < game.snakeCount+opp_snake_len;++j){
                        if(game.snakes[j].alive){

                            enemies.push_back(game.snakes[j]);
                            break;

                        }
                    }


                }
                else{
                    for(int j = 0;j < game.snakeCount;++j){
                        if (game.snakes[j].alive){

                            enemies.push_back(game.snakes[j]);
                            break;
                        }
                    }

                }*/


                //std::vector<float> input = encode_state_full(s, energy, enemies, game.w, game.h, 256);
                //double vnet = net.forward(input);


                int fl = floodFill(hx, hy, grcurp, game, min(4, s.len));
                if (fl < min(4, s.len) /*|| get_safe_moves(hx, hy, grcurp, game.w, game.h) == 0*/)sc += params.flood;

                //cerr << sc/200 << " " << vnet << endl;
                // 7️⃣ Backprop
                //sc = max(min(sc / 200.0, 1.0), 0.0);
                //vnet = max(min(vnet, 1.0), 0.0);



                backprop(node[i], (sc / 200.0));

                //}

            }


            ++turn;


        }

        cerr << "TURN=" << turn << " " << "node=" << nodeCount << ", child=" << childCount << ", DEPTH=" << DEPTH << endl;

        string ans;

        //direction
        for (int i = 0; i < my_snake_len; ++i) {
            int indc = -1;
            double maxi = -2e9;
            cerr << my_id_snake[i] << endl;
            indc = bestMove(root[i]);


            if (indc != -1) {
                ans += to_string(my_id_snake[i]) + " " + direction[indc] + ";";
            }
            else {
                ans += "WAIT;";
            }

        }

        if (ans.empty())cerr << "empty ans" << endl;

        return ans;


    }

    int compute_max_gap(
        int tx, int ty,
        const vector<vector<pair<int, int>>>& parent,
        const Grid& grid,
        int W, int H, int& d
    ) {
        int max_gap = 0;
        int current_gap = 0;

        int x = tx, y = ty;

        while (x != -1 && y != -1) {
            d++;
            int ny = y + 1;

            bool no_support = false;

            if (ny >= H) {
                no_support = true;
            }
            else {
                if (!(grid.cell[ny][x] & (WALL | ENERGY | SNAKE))) {
                    no_support = true;
                }
            }

            if (grid.cell[y][x] & SNAKE) {
                d += 5;
                //break;
            }

            if (no_support) {
                current_gap++;
                if (current_gap > max_gap)
                    max_gap = current_gap;
            }
            else {
                current_gap = 0; // 🔥 reset dès qu’il y a support
            }

            auto [px, py] = parent[y][x];
            x = px;
            y = py;
        }

        return max_gap;
    }

    bool somethingSolidUnder(GameState& g, int x, int y,
        const unordered_set<long long>& meta,
        int total) {

        int ny = y + 1;

        // --- hors map = solide
        if (x < 0 || x >= g.w || ny < 0)
            return true;

        // --- sol
        if (ny >= g.h)
            return true;

        // --- ignore groupe (clé = x + y*W)
        long long key = (long long)x << 32 | ny;
        if (meta.count(key))
            return false;

        // --- mur
        if (g.grid.cell[ny][x] & WALL)
            return true;

        // --- énergie
        if (g.grid.cell[ny][x] & ENERGY)
            return true;

        // --- autres snakes
        for (int i = 0; i < total; i++) {
            Snake& s = g.snakes[i];
            if (!s.alive) continue;

            for (int k = 0; k < s.len; k++) {
                int idx = (s.tail + k) % MAX_BODY;
                Pos p = s.body[idx];

                if (p.x == x && p.y == ny)
                    return true;
            }
        }

        return false;
    }

    bool touchingVertical(const Snake& a, const Snake& b) {
        for (int i = 0; i < a.len; i++) {
            int id1 = (a.tail + i) % MAX_BODY;
            Pos p1 = a.body[id1];

            for (int j = 0; j < b.len; j++) {
                int id2 = (b.tail + j) % MAX_BODY;
                Pos p2 = b.body[id2];

                if (p1.x == p2.x && abs(p1.y - p2.y) == 1)
                    return true;
            }
        }
        return false;
    }

    vector<vector<int>> getGroupsor(GameState& g, int total) {
        vector<vector<int>> groups;
        vector<bool> visited(total, false);

        for (int i = 0; i < total; i++) {
            if (visited[i] || !g.snakes[i].alive) continue;

            vector<int> group;
            stack<int> st;
            st.push(i);

            while (!st.empty()) {
                int u = st.top(); st.pop();
                if (visited[u]) continue;

                visited[u] = true;
                group.push_back(u);

                for (int v = 0; v < total; v++) {
                    if (v == u || visited[v]) continue;
                    if (!g.snakes[v].alive) continue;

                    if (touchingVertical(g.snakes[u], g.snakes[v]))
                        st.push(v);
                }
            }

            groups.push_back(group);
        }

        return groups;
    }

    vector<vector<int>> getGroups(GameState& g, int total) {
        vector<vector<int>> groups;
        vector<bool> visited(total, false);

        for (int i = 0; i < total; ++i) {
            Snake& si = g.snakes[i];
            if (visited[i] || !si.alive) continue;

            vector<int> group;
            stack<int> st;
            st.push(i);

            while (!st.empty()) {
                int u = st.top(); st.pop();
                if (visited[u]) continue;

                visited[u] = true;
                group.push_back(u);

                Snake& su = g.snakes[u];
                for (int v = 0; v < total; ++v) {
                    if (visited[v] || v == u) continue;
                    Snake& sv = g.snakes[v];
                    if (!sv.alive) continue;

                    if (touchingVertical(su, sv)) {
                        st.push(v);
                    }
                }
            }

            groups.emplace_back(move(group)); // pas de copie
        }

        return groups;
    }

    void doFalls(GameState& g, int total) {

        while (true) {
            bool somethingFell = false;

            auto groups = getGroups(g, total);

            for (auto& grp : groups) {

                // --- meta body
                unordered_set<long long> meta;

                for (int i : grp) {
                    Snake& s = g.snakes[i];

                    for (int k = 0; k < s.len; k++) {
                        int idx = (s.tail + k) % MAX_BODY;
                        Pos p = s.body[idx];

                        long long key = ((long long)p.x << 32) | (unsigned int)p.y;
                        meta.insert(key);
                    }
                }

                // --- check fall
                bool canFall = true;

                for (auto key : meta) {
                    int x = key >> 32;
                    int y = (int)key;

                    if (somethingSolidUnder(g, x, y, meta, total)) {
                        canFall = false;
                        break;
                    }
                }

                // --- apply fall
                if (canFall) {
                    somethingFell = true;

                    for (int i : grp) {
                        Snake& s = g.snakes[i];

                        for (int k = 0; k < s.len; k++) {
                            int idx = (s.tail + k) % MAX_BODY;
                            s.body[idx].y += 1;
                        }

                        // --- death
                        bool dead = true;
                        for (int k = 0; k < s.len; k++) {
                            int idx = (s.tail + k) % MAX_BODY;

                            if (s.body[idx].y < g.h + 1) {
                                dead = false;
                                break;
                            }
                        }

                        if (dead)
                            s.alive = false;
                    }
                }
            }

            if (!somethingFell) break;
        }
    }

    bool isSnakeSupportedOrDead(GameState& g, Snake& s)
    {
        bool hasOutside = false;
        bool hasSupportInside = false;

        for (int k = 0; k < s.len; k++) {
            int id = (s.tail + k) % MAX_BODY;
            Pos c = s.body[id];

            // sort sur les côtés
            /*if(c.x < 0 || c.x >= g.w){
                hasOutside = true;
                continue;
            }*/

            // dans la grille
            if (c.y >= 0 && c.y < g.h) {

                // sol
                if (c.x >= 0 && c.x < g.w) {
                    hasSupportInside = true;
                }
                else if (c.y + 1 >= 0 && c.y + 1 < g.h) {
                    if (g.grid.cell[c.y + 1][c.x] & (WALL | ENERGY)) {
                        hasSupportInside = true;
                    }
                }
            }
        }

        // 💀 condition de mort
        if (!hasSupportInside) {
            return false;
        }

        return true;
    }

    void doFalls2(GameState& g, int total_snakes) {

        vector<int> fallDist(total_snakes, 0);
        vector<bool> alive(total_snakes, true);

        vector<int> airborne;
        vector<int> grounded;

        // init airborne = tous les snakes vivants
        for (int i = 0; i < total_snakes; i++) {
            if (g.snakes[i].alive)
                airborne.push_back(i);
        }

        while (true) {

            bool somethingFell = false;

            // --- propagation grounded ---
            bool changed = true;
            while (changed) {
                changed = false;

                for (int idx = 0; idx < (int)airborne.size(); idx++) {
                    int i = airborne[idx];
                    Snake& s = g.snakes[i];

                    bool isGrounded = false;

                    for (int k = 0; k < s.len; k++) {
                        int id = (s.tail + k) % MAX_BODY;
                        Pos c = s.body[id];

                        // sol
                        if (c.y + 1 >= g.h - 1) {
                            isGrounded = true;
                            break;
                        }

                        // mur / pomme
                        if (g.grid.cell[c.y + 1][c.x] & (WALL | ENERGY)) {
                            isGrounded = true;
                            break;
                        }

                        // touche snake grounded
                        for (int j : grounded) {
                            Snake& os = g.snakes[j];
                            for (int kk = 0; kk < os.len; kk++) {
                                int id2 = (os.tail + kk) % MAX_BODY;
                                Pos p2 = os.body[id2];

                                if (p2.x == c.x && p2.y == c.y + 1) {
                                    isGrounded = true;
                                    break;
                                }
                            }
                            if (isGrounded) break;
                        }
                        if (isGrounded) break;
                    }

                    if (isGrounded) {
                        grounded.push_back(i);
                        airborne[idx] = airborne.back();
                        airborne.pop_back();
                        idx--;
                        changed = true;
                    }
                }
            }

            // --- chute ---
            for (int i : airborne) {
                Snake& s = g.snakes[i];

                bool canFall = true;

                for (int k = 0; k < s.len; k++) {
                    int id_ = (s.tail + k) % MAX_BODY;
                    Pos& c = s.body[id_];

                    // touche le sol (hors grille)
                    if (c.y + 1 >= g.h - 1) {
                        canFall = false;
                        break;
                    }

                    // collision avec mur ou énergie
                    if (/*c.x >= 0 && c.x < g.w &&*/ c.y + 1 >= 0 && c.y + 1 < g.h) {
                        if (g.grid.cell[c.y + 1][c.x] & (WALL | ENERGY)) {
                            canFall = false;
                            break;
                        }
                    }
                }

                if (!canFall) {
                    continue;
                }

                somethingFell = true;

                for (int k = 0; k < s.len; k++) {
                    int id = (s.tail + k) % MAX_BODY;
                    s.body[id].y += 1;
                }

                fallDist[i]++;

                /*if(!isSnakeSupportedOrDead(g, s)){
                    g.snakes[i].alive = false;
                    continue;
                }*/


                // out of bounds
                bool dead = true;
                for (int k = 0; k < s.len; k++) {
                    int id = (s.tail + k) % MAX_BODY;
                    if (s.body[id].y <= g.h) {
                        dead = false;
                        break;
                    }
                }

                if (dead) {
                    s.alive = false;
                }
            }

            if (!somethingFell) break;
        }

        for (int i = 0; i < total_snakes; i++) {
            if (g.snakes[i].alive) {
                // 💀 check support horizontal
                if (!isSnakeSupportedOrDead(g, g.snakes[i])) {
                    g.snakes[i].alive = false;

                }

            }

        }

    }

    bool touchingVertical2(const Snake& a, const Snake& b) {
        for (int i = 0; i < a.len; i++) {
            int id1 = (a.tail + i) % MAX_BODY;
            Pos p1 = a.body[id1];

            for (int j = 0; j < b.len; j++) {
                int id2 = (b.tail + j) % MAX_BODY;
                Pos p2 = b.body[id2];

                if (abs(p1.x - p2.x) == 0 && abs(p1.y - p2.y) == 1)
                    return true;
            }
        }
        return false;
    }

    vector<vector<int>> getIntercoiled(GameState& g, int total) {

        vector<vector<int>> groups;
        vector<bool> vis(total, false);

        for (int i = 0; i < total; i++) {
            if (vis[i] || !g.snakes[i].alive) continue;

            vector<int> group;
            queue<int> q;
            q.push(i);

            while (!q.empty()) {
                int u = q.front(); q.pop();
                if (vis[u]) continue;

                vis[u] = true;
                group.push_back(u);

                for (int v = 0; v < total; v++) {
                    if (u == v || vis[v]) continue;
                    if (!g.snakes[v].alive) continue;

                    if (touchingVertical2(g.snakes[u], g.snakes[v]))
                        q.push(v);
                }
            }

            if (group.size() > 1)
                groups.push_back(group);
        }

        return groups;
    }

    void doIntercoiledFalls(GameState& g, int total) {

        while (true) {

            bool fell = false;

            auto groups = getIntercoiled(g, total);

            for (auto& grp : groups) {

                bool canFall = true;

                for (int id : grp) {
                    Snake& s = g.snakes[id];

                    for (int k = 0; k < s.len; k++) {
                        int idx = (s.tail + k) % MAX_BODY;
                        Pos c = s.body[idx];

                        if (c.y + 1 >= g.h) {
                            canFall = false;
                            break;
                        }

                        if (g.grid.cell[c.y + 1][c.x] & WALL) {
                            canFall = false;
                            break;
                        }
                    }

                    if (!canFall) break;
                }

                if (canFall) {
                    fell = true;

                    for (int id : grp) {
                        Snake& s = g.snakes[id];

                        for (int k = 0; k < s.len; k++) {
                            int idx = (s.tail + k) % MAX_BODY;
                            s.body[idx].y += 1;
                        }
                    }
                }
            }

            if (!fell) break;
        }
    }

    int get_safe_moves(int hx, int hy, const Grid& grid, int w, int h) {
        // tête du snake
        int count = 0;

        // DIRS : UP, DOWN, LEFT, RIGHT
        const int DX[4] = { 0, 0, -1, 1 };
        const int DY[4] = { -1, 1, 0, 0 };

        for (int i = 0; i < 4; ++i) {
            int nx = hx + DX[i];
            int ny = hy + DY[i];

            // vérifier qu'on reste dans la grille
            if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                continue;

            // vérifier collision mur ou snake
            if (grid.cell[ny][nx] & (WALL | SNAKE))
                continue;


            count++;
        }

        return count;
    }



    void backpropor(int node, double result) {

        while (node != -1) {

            nodes[node].visits++;
            nodes[node].score += result;

            node = nodes[node].parent;
        }
    }

    // Optimisation de backpropagation
    // nodes : tableau contigu de Node
    // node : indice de départ
    // result : score à propager
    void backprop(int start_node, double result) {
        int node = start_node;

        // Boucle simple sur l'arbre
        while (node != -1) {
            Node& n = nodes[node];

            // Accès direct aux champs pour réduire overhead
            n.visits += 1;
            n.score += result;

            // Préfetch du parent pour améliorer la localité mémoire
            int parent = n.parent;
            if (parent != -1) {
                __builtin_prefetch(&nodes[parent], 1, 3); // lecture + haute priorité
            }

            node = parent;
        }
    }


    int bestChildIndex(int rootNode) {
        int start = nodes[rootNode].first_child;
        int bestChild = -1;
        double bestScore = -1e30;

        for (int j = 0; j < nodes[rootNode].child_count; j++) {
            int c = children[start + j];  // l’indice réel de l’enfant
            double avg = nodes[c].visits ? nodes[c].score / (double)nodes[c].visits : 0;
            //avg *= nodes[c].mult;
            if (avg > bestScore) {
                bestScore = avg;
                bestChild = c;
            }

            cerr << nodes[c].move
                << " visits=" << nodes[c].visits
                << " score=" << nodes[c].score
                << " avg=" << fixed << setprecision(6) << avg
                << " mult" << nodes[c].mult
                << endl;

            //cerr << j << "=" << fixed << setprecision(5) << avg << endl;
        }

        return bestChild; // retourne l’indice réel dans nodes[]
    }

    // Pour récupérer la direction finale
    int bestMove(int rootNode) {
        int c = bestChildIndex(rootNode);
        if (c != -1) return nodes[c].move; // utilise la move stockée
        return -1; // pas de coup valide
    }





    int applyGravity(GameState& g, Snake& s, Grid& gr)
    {
        int fall = INT_MAX;

        // retirer le snake de la grille
        /*for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];
            if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                gr.cell[p.y][p.x] &= ~SNAKE;
        }*/

        // calcul de la distance maximale de chute
        for (int i = 0; i < s.len; i++) {
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];

            int d = 0;
            int y = p.y;

            while (true) {
                y++;

                if (y >= g.h) break;

                //if(gr.cell[y][p.x] != EMPTY)
                //    break;

                if (gr.cell[y][p.x] & (WALL | SNAKE | ENERGY))
                    break;

                d++;
            }

            fall = std::min(fall, d);
        }

        if (fall == 0)
            return INT_MAX;

        // appliquer la chute
        for (int i = 0; i < s.len; i++) {
            int id = (s.tail + i) % MAX_BODY;
            s.body[id].y += fall;
        }

        return fall;

        // remettre dans la grille
        /*for(int i = 0; i < s.len; i++){
            int id = (s.tail + i) % MAX_BODY;
            Pos p = s.body[id];

            if(p.y >=0 && p.y < g.h && p.x >=0 && p.x < g.w)
                gr.cell[p.y][p.x] |= SNAKE;
        }*/
    }


    int floodFill(int sx, int sy, Grid& gr, GameState& g, int limit = 30) {

        bool vis[50][50] = {};
        queue<pair<int, int>> q;

        q.push({ sx, sy });
        vis[sy][sx] = true;

        int cnt = 0;

        while (!q.empty() && cnt < limit) {
            auto [x, y] = q.front(); q.pop();
            cnt++;

            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d];
                int ny = y + DY[d];

                if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
                if (vis[ny][nx]) continue;
                if (gr.cell[ny][nx] & (WALL | SNAKE)) continue;

                vis[ny][nx] = true;
                q.push({ nx, ny });
            }
        }

        return cnt;
    }


    Snake getSnake(int id, Snake my_snake[8], int len, int& ind) {
        for (int i = 0; i < len; ++i) {
            if (my_snake[i].id == id) {
                ind = i;
                return my_snake[i];
            }
        }


    }

    bool isVertical(Snake& s) {

        int x0 = s.body[s.head].x;

        for (int i = 0; i < s.len; i++) {

            int id = (s.tail + i) % MAX_BODY;

            if (s.body[id].x != x0)
                return false;
        }

        return true;
    }


};


void parseGrid(Grid& grid, int height, int width) {
    for (int y = 0; y < height; y++) {
        std::string row;
        std::getline(std::cin, row);
        cerr << row << endl;

        for (int x = 0; x < width; x++) {
            if (row[x] == '#') {
                grid.cell[y + BORDERH][x + BORDER] = WALL;
            }
            else {
                grid.cell[y + BORDERH][x + BORDER] = EMPTY;
            }
        }
    }
}

bool parseBody(const std::string& s, Snake& snake, Grid& grid) {
    snake.len = 0;
    snake.head = 0;
    snake.tail = 0;

    int x = 0, y = 0;
    int idx = 0;

    Pos temp[MAX_BODY];

    for (size_t i = 0; i < s.size(); ) {

        // --- parse x ---
        int sign = 1;
        if (s[i] == '-') {
            sign = -1;
            i++;
            //return false;
        }

        x = 0;
        while (i < s.size() && isdigit(s[i])) {
            x = x * 10 + (s[i] - '0');
            i++;
        }
        x *= sign;

        if (i < s.size() && s[i] == ',') i++;

        // --- parse y ---
        sign = 1;
        if (s[i] == '-') {
            sign = -1;
            i++;
            //return false;
        }

        y = 0;
        while (i < s.size() && isdigit(s[i])) {
            y = y * 10 + (s[i] - '0');
            i++;
        }
        y *= sign;

        temp[idx++] = { (int8_t)(x + BORDER), (int8_t)(y + BORDERH) };


        if (i < s.size() && s[i] == ':') i++;
    }

    snake.len = idx;

    // inverser pour avoir head à la fin (ton système circulaire)
    for (int i = 0; i < snake.len; i++) {
        snake.body[i] = temp[snake.len - 1 - i];
        Pos b = snake.body[i];
        //grid.cell[b.y][b.x] = SNAKE;
    }

    snake.head = snake.len - 1;
    snake.tail = 0;

    return true;
}

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
Grid grid;
SM simulation;

int main()
{
    srand(time(NULL));
    cerr << "FUCK UP" << endl;
    int my_id;
    cin >> my_id; cin.ignore();
    cerr << my_id << endl;
    int width;
    cin >> width; cin.ignore();
    cerr << width << endl;
    int height;
    cin >> height; cin.ignore();
    cerr << height << endl;


    parseGrid(grid, height, width);
    /*for (int i = 0; i < height; i++) {
        string row;
        getline(cin, row);

    }*/

    simulation.loadWeights();
    simulation.loadWeightsPUCT();

    map<int, int> snake_player;
    int snakebots_per_player;
    cin >> snakebots_per_player; cin.ignore();
    cerr << snakebots_per_player << endl;
    for (int i = 0; i < snakebots_per_player; i++) {
        int my_snakebot_id;
        cin >> my_snakebot_id; cin.ignore();
        cerr << my_snakebot_id << endl;
        snake_player[my_snakebot_id] = 0;
    }
    for (int i = 0; i < snakebots_per_player; i++) {
        int opp_snakebot_id;
        cin >> opp_snakebot_id; cin.ignore();
        cerr << opp_snakebot_id << endl;
        snake_player[opp_snakebot_id] = 1;
    }

    Snake last_my_snake[8];
    int last_my_snake_len = 0;

    int turn = 0;
    // game loop
    while (1) {
        Grid grid_g = grid;
        __builtin_memcpy(&grid_g, &grid, sizeof(Grid));
        Pos energy[MAX_POWER];
        int inde = 0;

        int power_source_count;
        cin >> power_source_count; cin.ignore();
        cerr << "power=" << power_source_count << endl;
        for (int i = 0; i < power_source_count; i++) {
            int x;
            int y;
            cin >> x >> y; cin.ignore();
            cerr << x << " " << y << endl;
            grid_g.cell[y + BORDERH][x + BORDER] = ENERGY;
            energy[i] = Pos{ (int8_t)(x + BORDER), (int8_t)(y + BORDERH) };

        }

        //if(turn == 0){
        auto startm = high_resolution_clock::now();

        simulation.distg = simulation.compute_distance_mapg(width + BORDER * 2, height + BORDERH, grid, energy, power_source_count, simulation.parentg_sim);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - startm);
        cerr << "DURATION=" << duration.count() << endl;
        int DURATION = duration.count();
        //}

        Snake my_snake[8], opp_snake[8];
        int my_id_snake[8];
        int indids = 0;
        int my_snake_len = 0, opp_snake_len = 0;

        int snakebot_count;
        cin >> snakebot_count; cin.ignore();
        cerr << snakebot_count << endl;
        for (int i = 0; i < snakebot_count; i++) {
            int snakebot_id;
            string body;
            cin >> snakebot_id >> body; cin.ignore();
            cerr << snakebot_id << " " << body << endl;

            Snake snake;
            snake.id = snakebot_id;
            snake.alive = true;
            bool good = parseBody(body, snake, grid_g);
            if (good) {
                if (snake_player[snakebot_id] == 0) {
                    my_snake[my_snake_len] = snake;
                    my_snake_len++;
                    my_id_snake[indids] = snakebot_id;
                    indids++;
                }
                else if (snake_player[snakebot_id] == 1) {
                    opp_snake[opp_snake_len] = snake;
                    opp_snake_len++;
                }
            }

        }

        /*for(int y = 0; y < height; y++) {

            for(int x = 0; x < width; x++) {
                if(grid_g.cell[y][x] & WALL) {
                    cerr << "#";
                }
                else if(grid_g.cell[y][x] & SNAKE) {
                    cerr << "O";
                }
                else if(grid_g.cell[y][x] & ENERGY) {
                    cerr << "e";
                } else {
                    cerr << " ";
                }
            }
            cerr << endl;
        }*/

        /*cerr << "len=" << 45.0/(double)my_snake_len << endl;
        string ans;
        for(int player = 0;player < my_snake_len;++player){
            ans += simulation.PlayB(player, height, width, grid_g, my_snake, my_snake_len, opp_snake, opp_snake_len, my_id_snake, energy, power_source_count, 40.0/(double)my_snake_len);

            // Write an action using cout. DON'T FORGET THE "<< endl"
            // To debug: cerr << "Debug messages..." << endl;


        }*/

        int time = 60;
        if (turn > 0)time = 45;

        Snake my_snakem[8], my_snakeb[8];
        int my_id_snakeb[8], my_id_snakem[8];
        int my_snake_lenb = 0, my_snake_lenm = 0;

        int mid = max(1, (my_snake_len / 2));

        cerr << "LEN=" << my_snake_len << endl;

        for (int i = 0; i < my_snake_len; ++i) {
            my_snakeb[i] = my_snake[i];
            my_snake_lenb++;
            my_id_snakeb[i] = my_id_snake[i];
        }

        /*for(int i = mid;i < my_snake_len;++i){
            opp_snake[opp_snake_len] = my_snake[i];
            opp_snake_len++;

        }*/

        int j = 0;
        for (int i = 0; i < my_snake_len; ++i) {
            my_snakem[j] = my_snake[i];
            my_snake_lenm++;
            my_id_snakem[j] = my_id_snake[i];
            ++j;
        }


        string ans = simulation.Play(height, width, grid_g, my_snake, my_snake_len, opp_snake, opp_snake_len, my_id_snake, energy, power_source_count, time - DURATION);
        //string ans2 = simulation.PlayMCTS(height, width, grid_g, my_snakem, my_snake_lenm, opp_snake, opp_snake_len, my_id_snakem, energy, power_source_count, time);

        cout << ans << endl;
        //cout << ans << ans2 << endl;
        cout.flush();

        /*last_my_snake_len = my_snake_lenb;
        for(int i = 0;i < my_snake_lenb;++i){
            last_my_snake[i] = my_snakeb[i];
        }*/

        ++turn;

    }
}