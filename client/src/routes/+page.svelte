<script>
  import '../app.css';

  let file = null;
  let imageUrl = null;
  let result = null;
  let analyzing = false;
  let query = "What's in the image"

  async function handleSubmit(event) {
    analyzing = true;

    event.preventDefault();

    if (file) {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://127.0.0.1:8000/upload-image/?query='+query, {
        method: 'POST',
        body: formData,
      });

      // Handle the response
      result = await response.json();

      analyzing = false;
      file = null;
    }
  }

  function handleFileChange(event) {
    file = event.target.files[0];
    if (file) {
      imageUrl = URL.createObjectURL(file);
    }
  }

  function handleFocus() {
    result = null;
    file = null;
    imageUrl = null;
  }

</script>

<div class="flex flex-col min-h-screen">
  <!-- Header -->
  <header class="bg-gray-800 text-white p-4">
    <h1>Functions with OpenAI</h1>
  </header>

  <!-- Main Content -->
  <main class="flex flex-1 overflow-auto">
    <!-- Left Column -->
    <div class="w-3/4 p-8 h-full">
      <!-- Rounded Pane with Header -->
      <div class="bg-white p-8 rounded-lg">
        <h2 class="text-xl font-bold mb-4">OpenAI Functions</h2>
        <!-- Quoted Reading in a Rounded Pane -->
        <div class="quote-pane bg-gray-100 p-6 rounded-lg">
          <form on:submit={handleSubmit}>
            <!-- New Query Input Field -->
            <label for="query" class="block font-medium text-gray-700">Are you looking for something in particular?</label>
            <input type="text" bind:value={query} on:focus={handleFocus} id="query" name="query" class="mt-1 block w-1/4 p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500" />

            <!-- File Input Field -->
            <label for="imageInput" class="block font-medium text-gray-700 mt-4">Select an image to analyze.</label>
            <input type="file" id="imageInput" accept="image/*" on:change={handleFileChange} class="mt-1 block w-full" />

            <!-- Submit Button -->
            <button type="submit" class="mt-4 px-4 py-2 bg-blue-500 text-white rounded">Analyze the image</button>
            {#if analyzing}
              <div>
                <b>Checking the image out...</b>
              </div>
            {/if}
          </form>
        </div>
      </div>

      <!-- Reading result area -->
      {#if result}
      <div class="bg-white p-8 mt-8 rounded-lg">
          <h2 class="text-xl font-bold mb-4">Here's the results.</h2>

          <!-- Analysis Area -->
          <div class="quote-pane bg-gray-100 p-6 rounded-lg flex items-start">
            <!-- Image Display -->
            {#if imageUrl}
              <img src={imageUrl} alt="Preview" class="max-w-xs mr-4" />
            {/if}

            <!-- Text Display -->
            <div>
              <p class="text-lg font-bold">{query}</p>
              <p>{result?.choices[0].message.content}</p>
            </div>
          </div>
      </div>
      {/if}

    </div>

  </main>

  <!-- Footer -->
  <footer class="bg-gray-800 text-white p-4">
    <p>Footer Content</p>
  </footer>
</div>

<style lang="postcss">
  :global(html) {
    background-color: theme(colors.gray.100);
  }
  .quote-pane {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
</style>
