import { useState, type ComponentProps } from 'react';
import '../../styles/search.css';

const HOT_WORDS = ['HELLO', 'THANK YOU', 'FAMILY', 'LEARN', 'FRIEND'];

function Search() {
  const [keyword, setKeyword] = useState('');
  const [selectedWord, setSelectedWord] = useState('HELLO');

  const handleSubmit: NonNullable<ComponentProps<'form'>['onSubmit']> = (event) => {
    event.preventDefault();
    const nextWord = keyword.trim();

    if (!nextWord) {
      return;
    }

    setSelectedWord(nextWord.toUpperCase());
  };

  const handleQuickSelect = (word: string) => {
    setKeyword(word);
    setSelectedWord(word);
  };

  return (
    <div className="search-page">
      <section className="search-shell">
        <div className="search-shell__glow search-shell__glow--left" />
        <div className="search-shell__glow search-shell__glow--right" />

        <div className="search-stage">
          <div className="search-column search-column--left">
            <form className="search-form" onSubmit={handleSubmit}>
              <label className="search-form__field" htmlFor="sign-search-input">
                <div className="search-form__control">
                  <input
                    id="sign-search-input"
                    type="text"
                    value={keyword}
                    onChange={(event) => setKeyword(event.target.value)}
                    placeholder="Try HELLO / THANK YOU / FAMILY"
                  />
                  <button type="submit">Search</button>
                </div>
              </label>
            </form>

            <div className="search-hotwords" aria-label="Popular sign words">
              {HOT_WORDS.map((word) => (
                <button
                  key={word}
                  type="button"
                  className={`search-hotwords__chip ${selectedWord === word ? 'is-active' : ''}`}
                  onClick={() => handleQuickSelect(word)}
                >
                  {word}
                </button>
              ))}
            </div>

            <section className="search-panel search-panel--model">
              <div className="model-preview">
                <div className="model-preview__frame">
                  <span className="search-card-tag">Model</span>
                  <div className="model-preview__grid" />
                  <div className="model-preview__hand">
                    <div className="model-preview__skeleton" aria-hidden="true">
                      <span className="joint joint--1" />
                      <span className="joint joint--2" />
                      <span className="joint joint--3" />
                      <span className="joint joint--4" />
                      <span className="joint joint--5" />
                      <span className="joint joint--6" />
                      <span className="joint joint--7" />
                      <span className="joint joint--8" />
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>

          <section className="search-panel search-panel--video">
            <div className="video-preview">
              <div className="video-preview__screen">
                <span className="search-card-tag">Video</span>
                <div className="video-preview__overlay" />
                <div className="video-preview__content">
                  <div className="video-preview__play" aria-hidden="true" />
                </div>
              </div>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}

export default Search;
