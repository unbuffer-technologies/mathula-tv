import pytest

from mathula_tv.zulu_numerals import AGREEMENT_CLASSES, cardinal, render


@pytest.mark.parametrize("n,cls,expected", [
    # the paper (Mahlaza, Magwenzi, Keet, Khumalo 2024), checked by two isiZulu grammarians
    (25, "2", "abangamashumi amabili nanhlanu"),
    (26, "2", "abangamashumi amabili nesithupha"),
    (14, "4", "emiyishumi nane"),
    (17, "4", "emiyishumi nesikhombisa"),
    (2, "2", "ababili"),                                     # ababili 'two people'
    (2, "10", "ezimbili"),                                   # izinja ezimbili 'two dogs'
    (5, "2", "abahlanu"),
    (6, "2", "abayisithupha"),                               # the -yisi- is there for 6-9
    # written in the news (counts in 8.4 million words)
    (3, "2", "abathathu"),                                   # 1,237
    (4, "2", "abane"),                                       # 684
    (10, "2", "abayishumi"),                                 # 164
    (2, "4", "emibili"),                                     # 2,659
    (3, "4", "emithathu"),                                   # 1,900
    (4, "4", "emine"),                                       # 1,023
    (5, "4", "emihlanu"),                                    # 1,452
    (2, "6", "amabili"),                                     # 4,669
    (5, "6", "amahlanu"),                                    # 814
    (6, "6", "ayisithupha"),                                 # 724
    (10, "6", "ayishumi"),                                   # 249
    (3, "10", "ezintathu"),                                  # 1,912
    (4, "10", "ezine"),                                      # 1,036
    (5, "10", "ezinhlanu"),                                  # 626
    (6, "10", "eziyisithupha"),                              # 753
    (10, "10", "eziyishumi"),                                # 315
    (20, "10", "ezingamashumi amabili"),
    (20, "6", "angamashumi amabili"),
    (4500, None, "izinkulungwane ezine namakhulu amahlanu"),  # the renderer's reading of R4,500
    (20, None, "amashumi amabili"),
    (500, None, "amakhulu amahlanu"),
    (1000, None, "inkulungwane"),
    (105, None, "ikhulu nanhlanu"),
    (11, None, "ishumi nanye"),
])
def test_cardinal_reproduces_the_paper_and_the_news(n, cls, expected):
    assert cardinal(n, cls) == expected


def test_remainders_after_na():
    assert [cardinal(10 + d, None) for d in range(1, 10)] == [
        "ishumi nanye", "ishumi nambili", "ishumi nantathu", "ishumi nane", "ishumi nanhlanu",
        "ishumi nesithupha", "ishumi nesikhombisa", "ishumi nesishiyagalombili", "ishumi nesishiyagalolunye"]


def test_only_the_first_unit_word_carries_the_agreement_after_na_come_the_plain_forms():
    # the model answered this way in all four attempts of the grammar test and the news agrees (namakhulu 38 against nangamakhulu 1); the paper's pattern 1d had (yi|ngama) here
    assert cardinal(3333, "10") == "eziyizinkulungwane ezintathu namakhulu amathathu namashumi amathathu nantathu"
    assert cardinal(1105, "2") == "abayinkulungwane nekhulu nanhlanu"
    assert cardinal(2000, "10") == "eziyizinkulungwane ezimbili"
    assert cardinal(1000, "2") == "abayinkulungwane" and cardinal(100, "2") == "abayikhulu"


def test_only_the_plural_counting_classes_and_the_range_1_to_9999_are_supported():
    assert AGREEMENT_CLASSES == ("2", "4", "6", "8", "10")
    for bad in (0, 10000):
        with pytest.raises(ValueError):
            cardinal(bad)
    with pytest.raises(ValueError):
        cardinal(5, "7")


def test_the_rendered_module_carries_the_counts_and_conventions():
    text = render()
    for needle in ("ababili 2,339", "148,000 bare digit", "abangamashumi amabili nanhlanu", "emiyishumi nane", "ngo-YEAR 6,558", "engu- 5,608", "eziyishumi ama- 0", "kunye", "ngomzuzu ka-90", "37,000 number words"):
        assert needle in text
