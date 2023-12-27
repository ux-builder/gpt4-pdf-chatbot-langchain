import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatPromptTemplate } from 'langchain/prompts';
import { RunnableSequence } from 'langchain/schema/runnable';
import { StringOutputParser } from 'langchain/schema/output_parser';
import type { Document } from 'langchain/document';
import type { VectorStoreRetriever } from 'langchain/vectorstores/base';

const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `당신은 오직 기업의 업무를 지원해주는 HR 전문가입니다.
당신은 기업 업무와 관련된 질문에만 답변하고 그외의 질문에 대해서는 정중하게 AI 인사쟁이 서비스 목적이 아니라고 답변해야 합니다. 
모든 질문에 대해 한국의 근로기준법과 관련 법령, 질의 회신 등 한국 기업에 근무하는 사람을 기준으로 답변해야 합니다.
답변은 최신의 법령을 기준으로 해야 합니다.
답변을 위해 계산을 해야 한다면 계산식을 반드시 제시하고 근거를 제공해야 합니다.
답변을 이해하기 쉽게 제공하기 위해 필요하다면 그림이나 도표를 제공해야 합니다.
답변을 하기 위해 질문자의 정보가 부족하다면 질문자에게 추가적인 정보를 요청해야 합니다.
답을 모르면 그냥 모른다고 말하세요. 답을 지어내려고 하지 마세요.
항상 한국어로 답변하세요.

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
Helpful answer in markdown:`;

const combineDocumentsFn = (docs: Document[], separator = '\n\n') => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

export const makeChain = (retriever: VectorStoreRetriever) => {
  const condenseQuestionPrompt =
    ChatPromptTemplate.fromTemplate(CONDENSE_TEMPLATE);
  const answerPrompt = ChatPromptTemplate.fromTemplate(QA_TEMPLATE);

  const model = new ChatOpenAI({
    temperature: 0.5, // increase temperature to get more creative answers
    maxTokens: 4096,
    modelName: 'gpt-4-1106-preview', //change this to gpt-4 if you have access
  });

  // Rephrase the initial question into a dereferenced standalone question based on
  // the chat history to allow effective vectorstore querying.
  const standaloneQuestionChain = RunnableSequence.from([
    condenseQuestionPrompt,
    model,
    new StringOutputParser(),
  ]);

  // Retrieve documents based on a query, then format them.
  const retrievalChain = retriever.pipe(combineDocumentsFn);

  // Generate an answer to the standalone question based on the chat history
  // and retrieved documents. Additionally, we return the source documents directly.
  const answerChain = RunnableSequence.from([
    {
      context: RunnableSequence.from([
        (input) => input.question,
        retrievalChain,
      ]),
      chat_history: (input) => input.chat_history,
      question: (input) => input.question,
    },
    answerPrompt,
    model,
    new StringOutputParser(),
  ]);

  // First generate a standalone question, then answer it based on
  // chat history and retrieved context documents.
  const conversationalRetrievalQAChain = RunnableSequence.from([
    {
      question: standaloneQuestionChain,
      chat_history: (input) => input.chat_history,
    },
    answerChain,
  ]);

  return conversationalRetrievalQAChain;
};
