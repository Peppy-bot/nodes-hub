import asyncio
import unittest

from my_python_robot_brain.frames import LatestValueMailbox


class LatestValueMailboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_consumer_receives_only_newest_pending_value(self):
        mailbox = LatestValueMailbox[int]()

        mailbox.offer(0)
        self.assertEqual(await mailbox.get(), 0)

        # The consumer is busy while many frames arrive.  The hand-off stays
        # bounded and the next read skips the stale backlog.
        for frame_id in range(1, 1000):
            self.assertTrue(mailbox.offer(frame_id))

        self.assertEqual(await mailbox.get(), 999)

    async def test_close_wakes_a_parked_consumer(self):
        mailbox = LatestValueMailbox[bytes]()
        waiting = asyncio.create_task(mailbox.get())
        await asyncio.sleep(0)

        mailbox.close()

        self.assertIsNone(await asyncio.wait_for(waiting, timeout=1))

    async def test_close_delivers_pending_final_value_before_end(self):
        mailbox = LatestValueMailbox[str]()
        mailbox.offer("final")

        mailbox.close()

        self.assertEqual(await mailbox.get(), "final")
        self.assertIsNone(await mailbox.get())

    async def test_offer_after_close_is_ignored(self):
        mailbox = LatestValueMailbox[int]()
        mailbox.close()

        self.assertFalse(mailbox.offer(1))
        self.assertIsNone(await mailbox.get())


if __name__ == "__main__":
    unittest.main()
