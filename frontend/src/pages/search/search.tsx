import { useState, type FormEvent } from 'react';
import '../../styles/search.css';

const HOT_WORDS = ['HELLO', 'THANK YOU', 'FAMILY', 'LEARN', 'FRIEND'];

function Search() {
  const [keyword, setKeyword] = useState('');
  const [selectedWord, setSelectedWord] = useState('HELLO');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
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

        <div className="search-header">
          <div className="search-header__copy">
            <span className="search-kicker">Sign Search</span>
            <h1>Search a sign word and preview the model with its matching video.</h1>
            <p>
              The left card is reserved for Panda3D FBX rendering. After the model files are
              connected later, it can sync with the sign language video on the right.
            </p>
          </div>

          <form className="search-form" onSubmit={handleSubmit}>
            <label className="search-form__field" htmlFor="sign-search-input">
              <span className="search-form__label">Search Vocabulary</span>
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
        </div>

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

        <div className="search-stage">
          <section className="search-panel search-panel--model">
            <div className="search-panel__meta">
              <span className="search-panel__eyebrow">Model Focus</span>
              <h2>{selectedWord}</h2>
              <p>Panda3D / FBX preview area for detailed hand pose playback.</p>
            </div>

            <div className="model-preview">
              <div className="model-preview__frame">
                <div className="model-preview__grid" />
                <div className="model-preview__hand">
                  <span className="model-preview__badge">Panda3D Canvas</span>
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
                <div className="model-preview__caption">
                  FBX model viewport placeholder. This area is ready for later Panda3D embedding.
                </div>
              </div>
            </div>
          </section>

          <section className="search-panel search-panel--video">
            <div className="search-panel__meta">
              <span className="search-panel__eyebrow">Video Focus</span>
              <h2>{selectedWord} Video</h2>
              <p>Right-side full-size card for the corresponding sign language clip.</p>
            </div>

            <div className="video-preview">
              <div className="video-preview__screen">
                <div className="video-preview__overlay" />
                <div className="video-preview__content">
                  <span className="video-preview__tag">Video Player</span>
                  <div className="video-preview__play" aria-hidden="true" />
                  <p>Sign video close-up area</p>
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
